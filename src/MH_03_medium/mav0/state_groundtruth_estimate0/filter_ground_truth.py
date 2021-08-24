from pandas import read_csv

df = read_csv("data.csv")

df = df[['#timestamp', ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',
         ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']]

print(df.head())

df.to_csv("data.csv", index=False)
