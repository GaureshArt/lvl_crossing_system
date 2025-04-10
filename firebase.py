import pyrebase
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

ir1=17    #11
ir2=27   #13
GPIO.setup(ir1,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(ir2,GPIO.IN,pull_up_down=GPIO.PUD_UP)


config={  "apiKey": "AIzaSyCBXqXNno05sFqwm5g4DZMHqeqHSoj1WSs",
          "authDomain": "levelcrossing.firebaseapp.com",
          "databaseURL": "https://levelcrossing-default-rtdb.firebaseio.com",
          "projectId": "levelcrossing",
          "storageBucket": "levelcrossing.firebasestorage.app"
          
        }

firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage=firebase.storage()
while True:
    irstate1=GPIO.input(ir1)
    irstate2=GPIO.input(ir2)
    time_hhmmss = time.strftime('%H:%M:%S')
    date_mmddyyyy = time.strftime('%d/%m/%Y')
    data = {"Date": date_mmddyyyy,"Time": time_hhmmss, "ir1": irstate1, "ir2": irstate2}
    print(f"Date:{date_mmddyyyy} Time:{time_hhmmss} ir1:{irstate1} ir2:{irstate2}")
    db.child("/message").push(data)

    time.sleep(1)