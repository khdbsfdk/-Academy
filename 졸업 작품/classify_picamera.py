# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera

import cv2
import RPi.GPIO as GPIO

from PIL import Image
from tflite_runtime.interpreter import Interpreter

STOP  = 0
FORWARD  = 1
BACKWORD = 2

 
CH1 = 0
CH2 = 1

  
OUTPUT = 1
INPUT = 0

  
HIGH = 1
LOW = 0

ENA = 26  #37 pin
ENB = 0   #27 pin

IN1 = 19  #35 pin
IN2 = 13  #33 pin
IN3 = 6   #31 pin
IN4 = 5   #29 pin

def setPinConfig(EN, INA, INB):
  GPIO.setup(EN, GPIO.OUT)
  GPIO.setup(INA, GPIO.OUT)
  GPIO.setup(INB, GPIO.OUT)
  # 100khz 로 PWM 동작 시킴 
  pwm = GPIO.PWM(EN, 100) 
  # 우선 PWM 멈춤.   
  pwm.start(0) 
  return pwm

# 모터 제어 함수
def setMotorContorl(pwm, INA, INB, speed, stat):
  #모터 속도 제어 PWM
  pwm.ChangeDutyCycle(speed)  
        
  if stat == FORWARD:
    GPIO.output(INA, HIGH)
    GPIO.output(INB, LOW)
            
  #뒤로
  elif stat == BACKWORD:
    GPIO.output(INA, LOW)
    GPIO.output(INB, HIGH)
            
  #정지
  elif stat == STOP:
    GPIO.output(INA, LOW)
    GPIO.output(INB, LOW)

            
    # 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(ch, speed, stat):
  if ch == CH1:
    #pwmA는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
    setMotorContorl(pwmA, IN1, IN2, speed, stat)
  else:
    #pwmB는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
    setMotorContorl(pwmB, IN3, IN4, speed, stat)
  print(speed)

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}
    
    
def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  
  
def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  
  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
    
  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

GPIO.setmode(GPIO.BCM)
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)
  
  
def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args() 
  
  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model) 
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  
  cap = cv2.VideoCapture(-1)
  #영상넓이 
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
  #영상높이 
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  key_detect = 0
  times=1

  while (key_detect==0):
    ret,image_src =cap.read()

    frame_width=image_src.shape[1]
    frame_height=image_src.shape[0]

    cut_d=int((frame_width-frame_height)/2)
    crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

    image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

    start_time = time.time()
    if (times==1):
      results = classify_image(interpreter, image)
      elapsed_ms = (time.time() - start_time) * 1000
      label_id, prob = results[0]

      print(labels[label_id],prob)
    cv2.putText(crop_img,labels[label_id] + " " + str(round(prob,3)), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    
    times=times+1
    if (times>1):
      times=1

    cv2.imshow('Detecting....',crop_img)

    if(label_id == 0):
      setMotor(CH1, 30, FORWARD)
      null = int(30)
    elif(label_id == 1):
      setMotor(CH1, 0, FORWARD)
      null = 0
    elif(label_id == 2):
      setMotor(CH1, 20, FORWARD)
      null = 20
    elif(label_id == 3):
      setMotor(CH1, 50, FORWARD)
      null = 50
    elif(label_id == 4):
      setMotor(CH1, 70, FORWARD)
      null = 70
    elif(label_id == 5):
      setMotor(CH1, null, FORWARD)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()
  GPIO.cleanup()
  
  
      
if __name__ == '__main__':
  main()

