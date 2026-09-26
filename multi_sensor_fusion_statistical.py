import numpy as np
from filterpy.kalman import KalmanFilter

def run_trial(num_frames=200):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    kf.P *= 1000
    kf.Q = np.eye(4) * 0.01
    kf.x = np.array([0, 0, 0, 0])

    R_sensor1 = np.array([[5, 0], [0, 5]])
    R_sensor2 = np.array([[50, 0], [0, 50]])

    # track cumulative squared error for each method, to compare fairly
    err_sensor1 = 0.0
    err_sensor2 = 0.0
    err_fused = 0.0

    for i in range(num_frames):
        tx, ty = i, i 

        s1 = (tx + np.random.randn() * 1, ty + np.random.randn() * 1)
        s2 = (tx + np.random.randn() * 5, ty + np.random.randn() * 5)

        kf.predict()

        kf.R = R_sensor1
        kf.update(np.array(s1))

        kf.R = R_sensor2
        kf.update(np.array(s2))

        est_x, est_y = kf.x[0], kf.x[1]

        # squared distance from truth, for each method
        err_sensor1 += (s1[0] - tx) ** 2 + (s1[1] - ty) ** 2
        err_sensor2 += (s2[0] - tx) ** 2 + (s2[1] - ty) ** 2
        err_fused += (est_x - tx) ** 2 + (est_y - ty) ** 2

    # average error per frame
    return err_sensor1 / num_frames, err_sensor2 / num_frames, err_fused / num_frames


# run multiple independent trials and average the results,
# since a single run can still get "lucky" or "unlucky"
num_trials = 20
totals = np.zeros(3)

for trial in range(num_trials):
    s1_err, s2_err, fused_err = run_trial(num_frames=200)
    totals += np.array([s1_err, s2_err, fused_err])

avg_s1, avg_s2, avg_fused = totals / num_trials

print(f"Averaged over {num_trials} trials of 200 frames each:")
print(f"Sensor 1 (precise) avg squared error : {avg_s1:.2f}")
print(f"Sensor 2 (noisy)   avg squared error : {avg_s2:.2f}")
print(f"Fused              avg squared error : {avg_fused:.2f}")

if avg_fused < avg_s1:
    print("\n✅ Fusion beat even the BEST individual sensor on average.")
elif avg_fused < avg_s2:
    print("\n⚠️ Fusion beat the worse sensor, but not the better one.")
else:
    print("\n❌ Fusion did worse than both individually — check R tuning.")
