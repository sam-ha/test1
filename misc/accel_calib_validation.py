# this is the code to validate accel calibration routine

import numpy as np
from numpy.linalg import inv
import pandas as pd
from matplotlib import pyplot as plt
accel_ref_data_loc= r'/Users/samhajhashemi/Downloads/ref_accel_translated_static_6pos.csv'
file=pd.read_csv(accel_ref_data_loc)
ref_xn_x_raw=(file[file['pos_num']==1]['accel_x']).mean()
ref_xn_y_raw=(file[file['pos_num']==1]['accel_y']).mean()
ref_xn_z_raw=(file[file['pos_num']==1]['accel_z']).mean()
ref_xp_x_raw=(file[file['pos_num']==2]['accel_x']).mean()
ref_xp_y_raw=(file[file['pos_num']==2]['accel_y']).mean()
ref_xp_z_raw=(file[file['pos_num']==2]['accel_z']).mean()
ref_yn_x_raw=(file[file['pos_num']==3]['accel_x']).mean()
ref_yn_y_raw=(file[file['pos_num']==3]['accel_y']).mean()
ref_yn_z_raw=(file[file['pos_num']==3]['accel_z']).mean()
ref_yp_x_raw=(file[file['pos_num']==4]['accel_x']).mean()
ref_yp_y_raw=(file[file['pos_num']==4]['accel_y']).mean()
ref_yp_z_raw=(file[file['pos_num']==4]['accel_z']).mean()
ref_zn_x_raw=(file[file['pos_num']==5]['accel_x']).mean()
ref_zn_y_raw=(file[file['pos_num']==5]['accel_y']).mean()
ref_zn_z_raw=(file[file['pos_num']==5]['accel_z']).mean()
ref_zp_x_raw=(file[file['pos_num']==6]['accel_x']).mean()
ref_zp_y_raw=(file[file['pos_num']==6]['accel_y']).mean()
ref_zp_z_raw=(file[file['pos_num']==6]['accel_z']).mean()



bias_ref=np.mat([0.5*(ref_xp_x_raw+ref_xn_x_raw), 0.5*(ref_yp_y_raw+
                        ref_yn_y_raw), 0.5*(ref_zp_z_raw+ref_zn_z_raw)])
s_ref=np.mat([[ref_xp_x_raw-bias_ref[0,0], ref_yp_x_raw-bias_ref[0,0], ref_zp_x_raw]-bias_ref[0,0],
  [ref_xp_y_raw-bias_ref[0,1], ref_yp_y_raw-bias_ref[0,1], ref_zp_y_raw-bias_ref[0,1]],
  [ref_xp_z_raw-bias_ref[0,2], ref_yp_z_raw-bias_ref[0,2], ref_zp_z_raw-bias_ref[0,2]]])

accel_dut_data_loc= r'/Users/samhajhashemi/Downloads/dut_accel_translated_static_6pos.csv'
file=pd.read_csv(accel_dut_data_loc)
dut_xn_x_raw=(file[file['pos_num']==1]['accel_x']).mean()
dut_xn_y_raw=(file[file['pos_num']==1]['accel_y']).mean()
dut_xn_z_raw=(file[file['pos_num']==1]['accel_z']).mean()
dut_xp_x_raw=(file[file['pos_num']==2]['accel_x']).mean()
dut_xp_y_raw=(file[file['pos_num']==2]['accel_y']).mean()
dut_xp_z_raw=(file[file['pos_num']==2]['accel_z']).mean()
dut_yn_x_raw=(file[file['pos_num']==3]['accel_x']).mean()
dut_yn_y_raw=(file[file['pos_num']==3]['accel_y']).mean()
dut_yn_z_raw=(file[file['pos_num']==3]['accel_z']).mean()
dut_yp_x_raw=(file[file['pos_num']==4]['accel_x']).mean()
dut_yp_y_raw=(file[file['pos_num']==4]['accel_y']).mean()
dut_yp_z_raw=(file[file['pos_num']==4]['accel_z']).mean()
dut_zn_x_raw=(file[file['pos_num']==5]['accel_x']).mean()
dut_zn_y_raw=(file[file['pos_num']==5]['accel_y']).mean()
dut_zn_z_raw=(file[file['pos_num']==5]['accel_z']).mean()
dut_zp_x_raw=(file[file['pos_num']==6]['accel_x']).mean()
dut_zp_y_raw=(file[file['pos_num']==6]['accel_y']).mean()
dut_zp_z_raw=(file[file['pos_num']==6]['accel_z']).mean()

bias_dut=np.mat([0.5*(dut_xp_x_raw+dut_xn_x_raw), 0.5*(dut_yp_y_raw+
                        dut_yn_y_raw), 0.5*(dut_zp_z_raw+dut_zn_z_raw)])
s_dut=np.mat([[dut_xp_x_raw-bias_dut[0,0], dut_yp_x_raw-bias_dut[0,0], dut_zp_x_raw]-bias_dut[0,0],
  [dut_xp_y_raw-bias_dut[0,1], dut_yp_y_raw-bias_dut[0,1], dut_zp_y_raw-bias_dut[0,1]],
  [dut_xp_z_raw-bias_dut[0,2], dut_yp_z_raw-bias_dut[0,2], dut_zp_z_raw-bias_dut[0,2]]])

# plt.plot(file[file['pos_num']==4]['accel_z'])
# plt.show()


Accel_Scale_Skew=np.matmul(s_dut,inv(s_ref))
Cal_Accel_Scale_Skew=inv(Accel_Scale_Skew)
Cal_bias_dut=np.matmul(Cal_Accel_Scale_Skew,np.transpose(bias_dut))
print(Accel_Scale_Skew)
print(Cal_Accel_Scale_Skew)
print(Cal_bias_dut)
