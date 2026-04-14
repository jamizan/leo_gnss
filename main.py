from utils import csv_operations, ExtendedKalmanFilter2D
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def position_rmse(reference_df, estimate_df, ref_x_col, ref_y_col, est_x_col, est_y_col):
    # Compare 2D position trajectories with a single RMSE scalar.
    n = min(len(reference_df), len(estimate_df))
    if n == 0:
        return float('inf')

    rx = reference_df[ref_x_col].to_numpy()[:n]
    ry = reference_df[ref_y_col].to_numpy()[:n]
    ex = estimate_df[est_x_col].to_numpy()[:n]
    ey = estimate_df[est_y_col].to_numpy()[:n]
    return float(np.sqrt(np.mean((ex - rx) ** 2 + (ey - ry) ** 2)))


def apply_uniform_timestamps(df, dt=0.1):
    # Rebuild monotonic timestamps because source data contains repeated time values.
    out = df.copy()
    out['timestamps'] = np.arange(len(out), dtype=float) * dt
    return out


def tune_ekf(true_df, acc_df, leo_df, sample_size=3000):
    # The accelerometer data in this dataset is dominated by gravity (~9.87 m/s²)
    # and does not capture horizontal motion dynamics, so accel_gain is fixed at 0.
    # We only tune process_std (random-walk velocity noise) and measurement_std.
    accel_candidates = [10.0, 50.0, 100.0]
    meas_candidates = [12.0, 16.0, 24.0]
    # Gate must be permissive — the filter needs LEO updates when velocity changes fast.
    gate_candidates = [1000.0, 1e6]

    n = min(sample_size, len(true_df), len(acc_df), len(leo_df))
    true_sample = true_df.iloc[:n].reset_index(drop=True)
    acc_sample = acc_df.iloc[:n].reset_index(drop=True)
    leo_sample = leo_df.iloc[:n].reset_index(drop=True)
    split_idx = max(2, int(0.7 * n))

    true_train = true_sample.iloc[:split_idx].reset_index(drop=True)
    acc_train = acc_sample.iloc[:split_idx].reset_index(drop=True)
    leo_train = leo_sample.iloc[:split_idx].reset_index(drop=True)

    true_val = true_sample.iloc[split_idx:].reset_index(drop=True)
    acc_val = acc_sample.iloc[split_idx:].reset_index(drop=True)
    leo_val = leo_sample.iloc[split_idx:].reset_index(drop=True)

    best = None
    for process_std in accel_candidates:
        for meas_std in meas_candidates:
            for gate_threshold in gate_candidates:
                candidate = ExtendedKalmanFilter2D(
                    process_accel_std=process_std,
                    measurement_pos_std=meas_std,
                    default_dt=0.1,
                    min_dt=0.005,
                    max_dt=0.2,
                    gate_threshold=gate_threshold,
                    accel_gain=0.0,
                    accel_to_position=True
                )
                ekf_train, _ = candidate.run(acc_train, leo_train, compute_rts=False)
                ekf_val, _ = candidate.run(acc_val, leo_val, compute_rts=False)

                train_score = position_rmse(true_train, ekf_train, 'true_x', 'true_y', 'ekf_x', 'ekf_y')
                val_score = position_rmse(true_val, ekf_val, 'true_x', 'true_y', 'ekf_x', 'ekf_y')
                score = 0.7 * val_score + 0.3 * train_score

                if best is None or score < best['rmse']:
                    best = {
                        'process_accel_std': process_std,
                        'measurement_pos_std': meas_std,
                        'accel_gain': 0.0,
                        'gate_threshold': gate_threshold,
                        'train_rmse': train_score,
                        'val_rmse': val_score,
                        'rmse': score
                    }

    if best is None:
        raise RuntimeError('EKF tuning failed to produce any candidate result')

    return best


def main():
    filepath = Path(__file__).resolve().parent / 'dataset' / 'df_withLEO.csv'
    csv_op = csv_operations(filepath)

    true_df, acc_df, _, leo_df = csv_op.read_csv()

    sample_dt = 0.1
    true_df = apply_uniform_timestamps(true_df, dt=sample_dt)
    acc_raw_df = apply_uniform_timestamps(acc_df, dt=sample_dt)
    leo_raw_df = apply_uniform_timestamps(leo_df, dt=sample_dt)

    best = tune_ekf(true_df, acc_raw_df, leo_raw_df)

    ekf = ExtendedKalmanFilter2D(
        process_accel_std=best['process_accel_std'],
        measurement_pos_std=best['measurement_pos_std'],
        default_dt=sample_dt,
        min_dt=0.005,
        max_dt=0.2,
        gate_threshold=best['gate_threshold'],
        accel_gain=best['accel_gain'],
        accel_to_position=True
    )
    ekf_df, rts_df = ekf.run(acc_raw_df, leo_raw_df, compute_rts=True)

    

    def plot_trajectory_comparison(plot_n, filename):

        true_plot = true_df.iloc[:plot_n]
        leo_plot = leo_raw_df.iloc[:plot_n]
        ekf_plot = ekf_df.iloc[:plot_n]
        rts_plot = rts_df.iloc[:plot_n]

        plt.figure(figsize=(10, 8))
        plt.plot(true_plot['true_x'], true_plot['true_y'], label='True position', linewidth=2)
        plt.plot(leo_plot['LEO_x'], leo_plot['LEO_y'], label='LEO', alpha=0.8)
        plt.plot(ekf_plot['ekf_x'], ekf_plot['ekf_y'], label='EKF', linewidth=2)
        plt.plot(rts_plot['rts_x'], rts_plot['rts_y'], label='RTS', linewidth=2)

        plt.title('Trajectory Comparison: True vs LEO vs EKF vs RTS')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.axis('equal')

        plot_path = Path(__file__).resolve().parent / f'{filename}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to: {plot_path}')

    plot_trajectory_comparison(plot_n=200, filename='trajectory_comparison_200')
    plot_trajectory_comparison(plot_n=500, filename='trajectory_comparison_500')
    plot_trajectory_comparison(plot_n=1000, filename='trajectory_comparison_1000')
    csv_op.create_csv(ekf_df, rts_df, output_filename='dataset/ekf_results.csv')

if __name__ == "__main__":
    main()