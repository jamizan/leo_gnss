import csv
import pandas as pd
import numpy as np
from typing import List, Tuple
from filterpy.kalman import ExtendedKalmanFilter as FilterPyEKF

class csv_operations:
    def __init__(self, filename):
        self.filename = filename

    # Read CSV file
    def read_csv(self):
        df = pd.read_csv(self.filename)
        true_df, acc_df, gnss_df, leo_df = self._split_data(df)
        return true_df, acc_df, gnss_df, leo_df

    # Split the data into separate DataFrames
    def _split_data(self, df):
        self.true_df = df[['timestamps', 'true_lat', 'true_lon', 'true_x', 'true_y']]
        self.acc_df = df[['timestamps', 'ax', 'ay']]
        self.gnss_df = df[['timestamps', 'gnss_lat', 'gnss_lon', 'gnss_x', 'gnss_y']]
        self.leo_df = df[['timestamps', 'LEO_lat', 'LEO_lon', 'LEO_x', 'LEO_y']]

        return self.true_df, self.acc_df, self.gnss_df, self.leo_df
    
    def create_csv(self, ekf_df, rts_df, output_filename):
        # Align rows by index to avoid NaN from timestamp outer joins.
        n = min(200, len(self.true_df), len(self.leo_df), len(ekf_df), len(rts_df))

        true_block = self.true_df[['timestamps', 'true_lat', 'true_lon', 'true_x', 'true_y']].iloc[:n].reset_index(drop=True)
        leo_block = self.leo_df[['LEO_lat', 'LEO_lon', 'LEO_x', 'LEO_y']].iloc[:n].reset_index(drop=True)
        ekf_block = ekf_df[['ekf_x', 'ekf_y', 'ekf_vx', 'ekf_vy']].iloc[:n].reset_index(drop=True)
        rts_block = rts_df[['rts_x', 'rts_y', 'rts_vx', 'rts_vy']].iloc[:n].reset_index(drop=True)

        combined_df = pd.concat([true_block, leo_block, ekf_block, rts_block], axis=1)
        column_headers = combined_df.columns.tolist()
        top_header = [
            'TIMESTAMP', 'TRUE', '', '', '',
            'LEO', '', '', '',
            'EKF', '', '', '',
            'RTS', '', '', ''
        ]

        # Semicolon keeps values in separate cells for Excel locales using comma decimals.
        with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(top_header)
            writer.writerow(column_headers)
            for row in combined_df.itertuples(index=False, name=None):
                writer.writerow(row)

        print(f'CSV file saved to: {output_filename}')





class ExtendedKalmanFilter2D:
    def __init__(
        self,
        process_accel_std=1.0,
        measurement_pos_std=5.0,
        default_dt=0.1,
        min_dt=0.005,
        max_dt=0.2,
        gate_threshold=9.21,
        accel_gain=1.0,
        accel_to_position=False
    ):
        # state: [x, y, vx, vy]
        self.ekf = FilterPyEKF(dim_x=4, dim_z=2)
        self.ekf.x = np.zeros((4, 1), dtype=float)
        self.ekf.P = np.eye(4, dtype=float) * 10.0

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        self.process_accel_var = float(process_accel_std) ** 2
        self.R = np.diag([float(measurement_pos_std) ** 2, float(measurement_pos_std) ** 2])
        self.default_dt = float(default_dt)
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self.gate_threshold = float(gate_threshold)
        self.accel_gain = float(accel_gain)
        self.accel_to_position = bool(accel_to_position)
        self.ekf.R = self.R.copy()

    def reset(self, init_x, init_y, init_vx=0.0, init_vy=0.0):
        self.ekf.x = np.array([[init_x], [init_y], [init_vx], [init_vy]], dtype=float)
        # Large velocity uncertainty: filter must learn velocity from measurements.
        self.ekf.P = np.diag([50.0, 50.0, 10000.0, 10000.0])
        self.ekf.R = self.R.copy()

    def _transition_jacobian(self, dt):
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def _measurement_function(self, state):
        return np.array([[state[0, 0]], [state[1, 0]]], dtype=float)

    def _measurement_jacobian(self, _state):
        return self.H

    def _infer_timestamp_scale(self, timestamps):
        raw = np.diff(timestamps.astype(float))
        positive = raw[raw > 0.0]
        if positive.size == 0:
            return 1.0

        median_diff = float(np.median(positive))

        # Heuristic: infer source unit so dt is expressed in seconds.
        if median_diff >= 1e7:
            return 1e-9  # nanoseconds
        if median_diff >= 1e4:
            return 1e-6  # microseconds
        if median_diff >= 10.0:
            return 1e-3  # milliseconds
        return 1.0  # already seconds

    def _normalize_dt(self, delta_t, ts_scale, fallback_dt):
        dt = float(delta_t) * float(ts_scale)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = fallback_dt

        return float(np.clip(dt, self.min_dt, self.max_dt))

    def predict(self, ax, ay, dt):
        F = self._transition_jacobian(dt)
        if self.accel_to_position:
            B = np.array([
                [0.5 * dt ** 2, 0.0],
                [0.0, 0.5 * dt ** 2],
                [dt, 0.0],
                [0.0, dt]
            ])
        else:
            # Treat acceleration as a velocity prior so LEO remains primary for position.
            B = np.array([
                [0.0, 0.0],
                [0.0, 0.0],
                [dt, 0.0],
                [0.0, dt]
            ])
        Q = self.process_accel_var * (B @ B.T)

        # FilterPy EKF stubs model predict(u) as scalar-only; apply control input explicitly.
        u = (self.accel_gain * np.array([ax, ay], dtype=float)).reshape(2, 1)
        self.ekf.x = F @ self.ekf.x + B @ u
        self.ekf.F = F
        self.ekf.Q = Q
        self.ekf.P = F @ self.ekf.P @ F.T + Q

    def update(self, meas_x, meas_y):
        z = np.array([[float(meas_x)], [float(meas_y)]], dtype=float)
        innovation = z - self._measurement_function(self.ekf.x)
        S = self.H @ self.ekf.P @ self.H.T + self.ekf.R
        mahalanobis = (innovation.T @ np.linalg.pinv(S) @ innovation).item()

        if mahalanobis > self.gate_threshold:
            return False

        self.ekf.update(z, HJacobian=self._measurement_jacobian, Hx=self._measurement_function)
        return True

    def run(self, acc_df, leo_df, compute_rts=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        acc = acc_df[['timestamps', 'ax', 'ay']].reset_index(drop=True)
        leo = leo_df[['timestamps', 'LEO_x', 'LEO_y']].reset_index(drop=True)

        if acc.empty or leo.empty:
            empty_ekf = pd.DataFrame(columns=['timestamps', 'ekf_x', 'ekf_y', 'ekf_vx', 'ekf_vy'])
            empty_rts = pd.DataFrame(columns=['timestamps', 'rts_x', 'rts_y', 'rts_vx', 'rts_vy'])
            return empty_ekf, empty_rts

        first_x = float(leo.loc[0, 'LEO_x'])
        first_y = float(leo.loc[0, 'LEO_y'])
        # Start at the first LEO position with zero velocity. A large initial
        # covariance for the velocity states lets the filter converge quickly
        # through the first few LEO measurement updates.
        self.reset(first_x, first_y, 0.0, 0.0)
        ts_scale = self._infer_timestamp_scale(acc['timestamps'].to_numpy())

        output_rows = []
        filtered_states = []
        filtered_covs = []
        predicted_states = []
        predicted_covs = []
        transition_jacobians = []
        previous_ts = float(acc.loc[0, 'timestamps'])
        previous_valid_dt = self.default_dt

        for i in range(len(acc)):
            current_ts = float(acc.loc[i, 'timestamps'])
            dt = self.default_dt if i == 0 else self._normalize_dt(current_ts - previous_ts, ts_scale, previous_valid_dt)
            F = self._transition_jacobian(dt)

            ax = float(acc.loc[i, 'ax'])
            ay = float(acc.loc[i, 'ay'])
            self.predict(ax, ay, dt)
            predicted_states.append(self.ekf.x.copy())
            predicted_covs.append(self.ekf.P.copy())
            transition_jacobians.append(F)

            if i < len(leo):
                meas_x = leo.loc[i, 'LEO_x']
                meas_y = leo.loc[i, 'LEO_y']
                if pd.notna(meas_x) and pd.notna(meas_y):
                    self.update(meas_x, meas_y)

            filtered_states.append(self.ekf.x.copy())
            filtered_covs.append(self.ekf.P.copy())

            output_rows.append({
                'timestamps': current_ts,
                'ekf_x': float(self.ekf.x[0, 0]),
                'ekf_y': float(self.ekf.x[1, 0]),
                'ekf_vx': float(self.ekf.x[2, 0]),
                'ekf_vy': float(self.ekf.x[3, 0])
            })

            previous_ts = current_ts
            previous_valid_dt = dt

        ekf_df = pd.DataFrame(output_rows)
        if compute_rts:
            rts_df = self._run_rts_smoother(
                timestamps=ekf_df['timestamps'].to_numpy(),
                filtered_states=filtered_states,
                filtered_covs=filtered_covs,
                predicted_states=predicted_states,
                predicted_covs=predicted_covs,
                transition_jacobians=transition_jacobians
            )
        else:
            rts_df = pd.DataFrame(columns=['timestamps', 'rts_x', 'rts_y', 'rts_vx', 'rts_vy'])

        return ekf_df, rts_df

    def _run_rts_smoother(
        self,
        timestamps,
        filtered_states: List[np.ndarray],
        filtered_covs: List[np.ndarray],
        predicted_states: List[np.ndarray],
        predicted_covs: List[np.ndarray],
        transition_jacobians: List[np.ndarray]
    ) -> pd.DataFrame:
        n = len(filtered_states)
        if n == 0:
            return pd.DataFrame(columns=['timestamps', 'rts_x', 'rts_y', 'rts_vx', 'rts_vy'])

        smoothed_states = [state.copy() for state in filtered_states]
        smoothed_covs = [cov.copy() for cov in filtered_covs]

        for k in range(n - 2, -1, -1):
            P_f = filtered_covs[k]
            F_next = transition_jacobians[k + 1]
            P_pred_next = predicted_covs[k + 1]

            # Pseudo-inverse improves robustness if covariance is near-singular.
            Ck = P_f @ F_next.T @ np.linalg.pinv(P_pred_next)

            smoothed_states[k] = filtered_states[k] + Ck @ (smoothed_states[k + 1] - predicted_states[k + 1])
            smoothed_covs[k] = P_f + Ck @ (smoothed_covs[k + 1] - P_pred_next) @ Ck.T

        rows = []
        for i in range(n):
            rows.append({
                'timestamps': float(timestamps[i]),
                'rts_x': float(smoothed_states[i][0, 0]),
                'rts_y': float(smoothed_states[i][1, 0]),
                'rts_vx': float(smoothed_states[i][2, 0]),
                'rts_vy': float(smoothed_states[i][3, 0])
            })

        return pd.DataFrame(rows)
