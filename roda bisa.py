import RPi.GPIO as GPIO
import os

os.system('clear')
GPIO.setwarnings(False)

in1 = 24
in2 = 23
en_a = 25
in3 = 16
in4 = 26
en_b = 6

GPIO.setmode(GPIO.BCM)

GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en_a,GPIO.OUT)

GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en_b,GPIO.OUT)

power_a = GPIO.PWM(en_a,70)
power_a.start(50)
power_b = GPIO.PWM(en_b,70)
power_b.start(50)

GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)

print("P")
while(True):
	user = input("Masukkan Input: ").lower()
	os.system('clear')

	if user == 'w':
		GPIO.output(in1,GPIO.HIGH)
		GPIO.output(in2,GPIO.LOW)
		GPIO.output(in4,GPIO.LOW)
		GPIO.output(in3,GPIO.HIGH)
		print('Maju!')

	elif user == 's':
		GPIO.output(in1,GPIO.LOW)
		GPIO.output(in2,GPIO.HIGH)
		GPIO.output(in4,GPIO.HIGH)
		GPIO.output(in3,GPIO.LOW)
		print('Mundur!')
		
	elif user == 'd':
		GPIO.output(in1,GPIO.LOW)
		GPIO.output(in2,GPIO.LOW)
		GPIO.output(in4,GPIO.LOW)
		GPIO.output(in3,GPIO.HIGH)
		print('Kanan!')
		
	elif user == 'a':
		GPIO.output(in1,GPIO.HIGH)
		GPIO.output(in2,GPIO.LOW)
		GPIO.output(in4,GPIO.LOW)
		GPIO.output(in3,GPIO.LOW)
		print('Kiri!')

	elif user == 'b':
		GPIO.output(in1,GPIO.LOW)
		GPIO.output(in2,GPIO.LOW)
		GPIO.output(in3,GPIO.LOW)
		GPIO.output(in4,GPIO.LOW)
		print("Berhenti!")
	
	elif user == 'exit':
		print("Log Out!")
		break

	else:
		print("Ada ada ji itu")

GPIO.cleanup()