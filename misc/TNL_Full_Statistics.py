import os
import numpy as np
from matplotlib import pylab as plt
import scipy as sp
import scipy.interpolate
import pandas as pd
import csv
import re
class temperature_nonlinearity_check(object):

  def __init__(self, input_log_file_name):
    self.input_log_file_name = input_log_file_name
    self.ax_raw = []
    self.ay_raw = []
    self.az_raw = []
    self.gx_raw = []
    self.gy_raw = []
    self.gz_raw = []
    self.temperature_original=[]
    self.ax_raw_linear_fit_residual=[]
    self.ay_raw_linear_fit_residual = []
    self.az_raw_linear_fit_residual = []
    self.gx_raw_linear_fit_residual = []
    self.gy_raw_linear_fit_residual = []
    self.gz_raw_linear_fit_residual = []

  def read_input_log(self):
    match=False
    with open(self.input_log_file_name, 'r') as f:
      for line in f:
        if re.match("(.*)(Gyroscope|Accelerometer|IMU Temperature)(.*)"
                    "Samples(.*)", line):
          match=True
          parameter=line.split(' ')[-2]
        if re.match("(.*)(count)(.*)", line):
          match=False
        if match:
          if parameter == 'Gyroscope':
            try:
              self.gx_raw.append(float(line.split('      ')[1]))
              self.gy_raw.append(float(line.split('      ')[2]))
              self.gz_raw.append(float(line.split('      ')[3]))
            except Exception:
              pass
          elif parameter == 'Accelerometer':
            try:
              self.ax_raw.append(float(line.split('      ')[1]))
              self.ay_raw.append(float(line.split('      ')[2]))
              self.az_raw.append(float(line.split('      ')[3]))
            except:
              pass
          elif parameter == 'Temperature':
            try:
              self.temperature_original.append(float(line.split('      ')[1]))
            except:
              pass
    return self.ax_raw, self.ay_raw, self.az_raw, self.gx_raw, self.gy_raw, \
           self.gz_raw, self.temperature_original

  def interpolate_temp_accel(self):
    x_old = np.arange(len(self.temperature_original))
    y_old = self.temperature_original
    x_new = np.linspace(x_old.min(), x_old.max(), len(self.ax_raw))
    self.temperature_interpld_acc = sp.interpolate.interp1d(x_old, y_old, kind='cubic')(x_new)
    return self.temperature_interpld_acc

  def interpolate_temp_gyro(self):
    x_old = np.arange(len(self.temperature_original))
    y_old = self.temperature_original
    x_new = np.linspace(x_old.min(), x_old.max(), len(self.gx_raw))
    self.temperature_interpld_gyro = sp.interpolate.interp1d(x_old, y_old, kind='cubic')(x_new)
    return self.temperature_interpld_gyro

  def smooth(self, a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    if (WSZ%2) == 0:
      WSZ+=1
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    smoothed_data=np.concatenate((start, out0, stop))
    return smoothed_data

  def smooth_all(self, window_size_ratio):
    self.ax_smoothed = self.smooth(self.ax_raw, int(len(self.ax_raw) / window_size_ratio))
    self.ay_smoothed = self.smooth(self.ay_raw, int(len(self.ay_raw) / window_size_ratio))
    self.az_smoothed = self.smooth(self.az_raw, int(len(self.az_raw) / window_size_ratio))
    self.gx_smoothed = self.smooth(self.gx_raw, int(len(self.gx_raw) / window_size_ratio))
    self.gy_smoothed = self.smooth(self.gy_raw, int(len(self.gy_raw) / window_size_ratio))
    self.gz_smoothed = self.smooth(self.gz_raw, int(len(self.gz_raw) / window_size_ratio))
    return self.ax_smoothed, self.ay_smoothed, self.az_smoothed, \
           self.gx_smoothed, self.gy_smoothed, self.gz_smoothed

  def poly_fit(self, x, y, degree, fitted_x_spacing=100):
    coeffs = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coeffs)
    fitted_x = np.linspace(np.min(x), np.max(x), fitted_x_spacing)
    fitted_y = polynomial(fitted_x)
    return coeffs, polynomial, fitted_x, fitted_y

  def linear_poly_fit_all(self):
    self.coeffs_ax_fit, _, self.ax_fitted_x, self.ax_fitted_y = self.poly_fit(
      self.temperature_interpld_acc, self.ax_smoothed, 1)
    self.coeffs_ay_fit, _, self.ay_fitted_x, self.ay_fitted_y = self.poly_fit(
      self.temperature_interpld_acc, self.ay_smoothed, 1)
    self.coeffs_az_fit, _, self.az_fitted_x, self.az_fitted_y = self.poly_fit(
      self.temperature_interpld_acc, self.az_smoothed, 1)
    self.coeffs_gx_fit, _, self.gx_fitted_x, self.gx_fitted_y = self.poly_fit(
      self.temperature_interpld_gyro, self.gx_smoothed, 1)
    self.coeffs_gy_fit, _, self.gy_fitted_x, self.gy_fitted_y = self.poly_fit(
      self.temperature_interpld_gyro, self.gy_smoothed, 1)
    self.coeffs_gz_fit, _, self.gz_fitted_x, self.gz_fitted_y = self.poly_fit(
      self.temperature_interpld_gyro, self.gz_smoothed, 1)
    return ()

  def linear_fit_residual(self, x, y, coeff_n1_0, coeff_n1_1):
    R1 = []
    R2 = []
    for i in range(len(x)):
      residual = y[i] - (coeff_n1_0 * x[i] + coeff_n1_1)
      R1.append(residual)
      R2.append(residual ** 2)
    residual= sum(R2)**0.5
    return residual

  def linear_fit_residual_all(self):
    self.residual_ax= self.linear_fit_residual(self.temperature_interpld_acc, self.ax_smoothed, self.coeffs_ax_fit[0], self.coeffs_ax_fit[1])
    self.residual_ay= self.linear_fit_residual(self.temperature_interpld_acc, self.ay_smoothed, self.coeffs_ay_fit[0], self.coeffs_ay_fit[1])
    self.residual_az= self.linear_fit_residual(self.temperature_interpld_acc, self.az_smoothed, self.coeffs_az_fit[0], self.coeffs_az_fit[1])
    self.residual_gx= self.linear_fit_residual(self.temperature_interpld_gyro, self.gx_smoothed, self.coeffs_gx_fit[0], self.coeffs_gx_fit[1])
    self.residual_gy= self.linear_fit_residual(self.temperature_interpld_gyro, self.gy_smoothed, self.coeffs_gy_fit[0], self.coeffs_gy_fit[1])
    self.residual_gz= self.linear_fit_residual(self.temperature_interpld_gyro, self.gz_smoothed, self.coeffs_gz_fit[0], self.coeffs_gz_fit[1])
    return self.residual_ax, self.residual_ay, self.residual_az, self.residual_gx, self.residual_gy, self.residual_gz

  def plot_results(self):
    fig, axs = plt.subplots(3, 3, figsize=(50, 30))
    axs = axs.ravel()
    params_to_plot = [self.ax_raw, self.ay_raw, self.az_raw, self.gx_raw, self.gy_raw, self.gz_raw, self.temperature_original]
    labels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'temperature']
    for i in np.arange(0, 9):
      if i <= 6:
        axs[i].plot(params_to_plot[i])
        axs[i].set_ylabel(labels[i], fontsize=15)
        axs[i].set_ylim(0.7 * min(params_to_plot[i]),
                        1.3 * max(params_to_plot[i]))
      elif i == 7:
        axs[i].plot(self.temperature_interpld_acc, self.ax_smoothed, '-',
                    self.temperature_interpld_acc, self.ay_smoothed, '-',
                    self.temperature_interpld_acc, self.az_smoothed, '-',
                    self.ax_fitted_x, self.ax_fitted_y, '-',
                    self.ay_fitted_x, self.ay_fitted_y, '-',
                    self.az_fitted_x, self.az_fitted_y, '-')
        axs[i].set_xlabel("Temperature", fontsize=15)
        axs[i].set_ylabel("Accelerometer_Temp_Fit", fontsize=15)
        axs[i].legend(('ax_smoothed', 'ay_smoothed', 'az_smoothed',
                       'ax_linear_fit', 'ay_linear_fit', 'az_linear_fit'))
      else:
        axs[i].plot(self.temperature_interpld_gyro, self.gx_smoothed, '-',
                    self.temperature_interpld_gyro, self.gy_smoothed, '-',
                    self.temperature_interpld_gyro, self.gz_smoothed, '-',
                    self.gx_fitted_x, self.gx_fitted_y, '+',
                    self.gy_fitted_x, self.gy_fitted_y, '+',
                    self.gz_fitted_x, self.gz_fitted_y, '+')
        axs[i].set_xlabel("Temperature", fontsize=15)
        axs[i].set_ylabel("Gyroscope_Temp_Fit", fontsize=15)
        axs[i].legend(('gx_smoothed', 'gy_smoothed', 'gz_smoothed',
                       'gx_linear_fit', 'gy_linear_fit', 'gz_linear_fit'))

    plt.text(0.5, -0.25, 'Gyro_XYZ_linear_fit: %sx + %s, %sx + %s, %sx + %s'
             %("{0:.2E}".format(self.coeffs_gx_fit[0]), "{0:.2E}".format(self.coeffs_gx_fit[1]),
               "{0:.2E}".format(self.coeffs_gy_fit[0]), "{0:.2E}".format(self.coeffs_gy_fit[1]),
               "{0:.2E}".format(self.coeffs_gz_fit[0]), "{0:.2E}".format(self.coeffs_gz_fit[1])),
             size=15, ha='center', va='center', transform=axs[8].transAxes)

    plt.text(0.5, -0.35, 'Residual_Norm_Gyro_XYZ= %s, %s, %s' %("{0:.4f}".format(self.residual_gx),
    "{0:.4f}".format(self.residual_gy), "{0:.4f}".format(self.residual_gz)),
             size=15, ha='center', va='center', transform=axs[8].transAxes)

    plt.suptitle(self.input_log_file_name)
    plt.show()


if __name__ == "__main__":

  path ="/home/samhajhashemi/Downloads/test/attachments/raw/a15b12b4-22de-4120-9161-24fd75160ded/0_IMUTempCalibration_sns_cal_logs.txt"
  tnl = temperature_nonlinearity_check(path)
  ax, ay, az, gx, gy, gz, temperature_org = tnl.read_input_log()
  tnl.interpolate_temp_accel()
  tnl.interpolate_temp_gyro()
  tnl.smooth_all(15)
  tnl.linear_poly_fit_all()
  axr, ayr, azr, gxr, gyr, gzr = tnl.linear_fit_residual_all()
  print(axr, ayr, azr, gxr, gyr, gzr)
  tnl.plot_results()



"""
# path = "/home/samhajhashemi/Downloads/IMU/TNL/MP Gyro TCO FAILS/F2/"
# path = "/home/samhajhashemi/Downloads/IMU/TNL/passing/"
path = "/home/samhajhashemi/Downloads/test/attachments/raw/"

file_name=[]
residual_ax_full = []
residual_ay_full = []
residual_az_full = []
residual_gx_full = []
residual_gy_full = []
residual_gz_full = []
j = 0
for r, d, f in os.walk(path):
    for file in f:
        if re.match("(.*)(cal_logs)(.*)(.txt)", file):
          file_loc=os.path.join(r, file)
          # print(file_loc)
          tnl = temperature_nonlinearity_check(file_loc)
          ax, ay, az, gx, gy, gz, temperature_org = tnl.read_input_log()
          tnl.interpolate_temp_accel()
          tnl.interpolate_temp_gyro()
          tnl.smooth_all(15)
          tnl.linear_poly_fit_all()
          axr, ayr, azr, gxr, gyr, gzr = tnl.linear_fit_residual_all()
          residual_ax_full.append(axr)
          residual_ay_full.append(ayr)
          residual_az_full.append(azr)
          residual_gx_full.append(gxr)
          residual_gy_full.append(gyr)
          residual_gz_full.append(gzr)
          file_name.append(file_loc)
          j += 1
        if j>5000:
          break
          # print(axr, ayr, azr, gxr, gyr, gzr)
          # tnl.plot_results()
    else:
      continue
    break

output=pd.DataFrame([])
output['residual_ax_full']=residual_ax_full
output['residual_ay_full']=residual_ay_full
output['residual_az_full']=residual_az_full
output['residual_gx_full']=residual_gx_full
output['residual_gy_full']=residual_gy_full
output['residual_gz_full']=residual_gz_full
output['file_name']=file_name

output.to_csv(os.path.join(path, 'output.csv'))

plt.title('Accel_X',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_ax_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

plt.title('Accel_Y',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_ay_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

plt.title('Accel_Z',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_az_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

plt.title('Gyro_X',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_gx_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

plt.title('Gyro_Y',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_gy_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

plt.title('Gyro_Z',fontsize=20)
plt.xlabel('Linear_Residual',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.hist(residual_gz_full, bins=30, range=[0,max(residual_gz_full)*1.1], alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.show()

"""

"""
output=pd.read_csv('/home/samhajhashemi/Downloads/IMU/TNL/MP Gyro TCO FAILS/F2/output.csv')
# output=pd.read_csv('/home/samhajhashemi/Downloads/IMU/TNL/output.csv')
residual_ax_full = output.residual_ax_full
residual_ay_full = output.residual_ay_full
residual_az_full = output.residual_az_full
residual_gx_full = output.residual_gx_full
residual_gy_full = output.residual_gy_full
residual_gz_full = output.residual_gz_full
for param in [residual_ax_full, residual_ay_full, residual_az_full,
              residual_gx_full, residual_gy_full, residual_gz_full]:
  print("{0:.3f}".format(np.mean(param)), "{0:.3f}".format(np.std(param)),
        "{0:.3f}".format(np.mean(param)+np.std(param)*6),
        "{0:.3f}".format(np.min(param)), "{0:.3f}".format(np.max(param)))
stat="/home/samhajhashemi/Downloads/IMU/TNL/statistics.csv"


# with open(stat, 'w') as f:
#   for param in [residual_ax_full, residual_ay_full, residual_az_full,
#                 residual_gx_full, residual_gy_full, residual_gz_full]:
#     print("{0:.3f}".format(np.mean(param)),',', "{0:.3f}".format(np.std(param)),',',
#           "{0:.3f}".format(np.mean(param) + np.std(param) * 6),',',
#           "{0:.3f}".format(np.min(param)),',', "{0:.3f}".format(np.max(param)), file=f)
# print(len(residual_gx_full))
# print(len([i for i in residual_gx_full if i > 0.15]))
# print(len([i for i in residual_gy_full if i > 0.15]))
# print(len([i for i in residual_gz_full if i > 0.15]))

with open(stat, 'a+') as f:
  for i in np.arange(len(output.residual_gz_full)):
    if output.residual_gz_full[i]<0.15:
      print(output.file_name[i])
      f.write(output.file_name[i] + "\n")
      
print(output[output.residual_gz_full < 0.15].file_name[1])
f.close()
"""
