from utils.helpers import getBlobDiff

fullLeftFileName = r'C:\projects\tasks\#3532_Output_out_sample_sink_port_0_in0.ieb'
fullRightName =    r'C:\projects\tasks\#3380_Output_out_sample_sink_port_0_in0.ieb'
curMaxDiff = getBlobDiff(fullLeftFileName, fullRightName)
print(f'curMaxDiff={curMaxDiff}')