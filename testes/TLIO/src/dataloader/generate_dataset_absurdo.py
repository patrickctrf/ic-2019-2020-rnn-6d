from pandas import read_csv

# my timestamps
df = read_csv("imu_data.csv")

df = df[['#timestamp [ns]', ]]

print(df.head())

df.to_csv("my_timestamps_p.txt", index=False)

# imu measurements
df = read_csv("imu_data.csv")

df['zeros'] = 0
df['ones'] = 1

df = df[['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'zeros', 'zeros', 'zeros',
         'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]',
         'a_RS_S_z [m s^-2]', 'zeros', 'zeros', 'zeros', 'ones']]

print(df.head())

df.to_csv("imu_measurements.txt", index=False)

# evolving state
df = read_csv("ground_truth_data.csv")

df = df[['#timestamp', ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []',
         ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', ' v_RS_R_x [m s^-1]',
         ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']]

print(df.head())

df.to_csv("evolving_state.txt", index=False)

# calib state
df = read_csv("ground_truth_data.csv")

df['zeros'] = 0
df['ones'] = 1

df = df[['#timestamp', 'ones', 'zeros', 'zeros', 'zeros', 'ones', 'zeros',
         'zeros', 'zeros', 'ones', 'ones', 'zeros', 'zeros', 'zeros', 'ones', 'zeros',
         'zeros', 'zeros', 'ones', 'ones', 'zeros', 'zeros', 'zeros', 'ones', 'zeros',
         'zeros', 'zeros', 'ones', ' b_w_RS_S_x [rad s^-1]',
         ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]',
         ' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']]

print(df.head())

df.to_csv("evolving_state.txt", index=False)
