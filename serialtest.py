import serial
ser = serial.Serial('/dev/cu.usbmodem141401')  # open serial port
print(ser.name)         # check which port was really used
ser.write(b'hello')     # write a string
ser.close()             # close port