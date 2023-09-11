import pygame
import pyttsx3
import playsound
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from plyer import notification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# dataset
data = pd.DataFrame({
    'Age': [25, 30, 33, 45, 26],
    'Gender': ['F', 'F', 'M', 'F', 'M'],
    'Medication_Type': ['A', 'B', 'B', 'C', 'A'],
    'Adherence': [1, 1, 0, 1, 0],
    'Timestamp': ['01-09-2023', '02-09-2023', '03-09-2023', '04-09-2023', '05-09-2023']
})

# text-to-speech and initialize buzzer sound
engine = pyttsx3.init()
pygame.mixer.init()
vibration = pygame.mixer.Sound('vibration.wav')


# play sound and vibration
def play_sound(sound_file):
    playsound.playsound(sound_file)


def play_vibration():
    vibration.play()


# predict medication adherence with a neural network
def predict_medication_adherence(patient_data):
    x = data[['Age', 'Gender', 'Medication_Type']]
    y = data['Adherence']

    x = pd.get_dummies(x, columns=['Gender', 'Medication_Type'], drop_first=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    prediction = clf.predict(patient_data)

    return prediction


# Train model for medication adherence prediction
def model_medication_adherence(x_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_train = to_categorical(y_train, num_classes=2)
    model.fit(x_train, y_train, epochs=10, batch_size=16)

    return model


# reminder to take medication
def reminder_to_take_medication():
    current_time = datetime.now().strftime("%H:%M")
    medication_schedule = {
        "08:00": "Morning medication",
        "13:00": "Afternoon medication",
        "19:00": "Evening medication",
    }
    if current_time in medication_schedule:
        # simulate patient data
        patient_data = pd.DataFrame({
            'Age': [45],
            'Gender_F': [1],
            'Medication_Type_C': [1]
        })
        predictions = model_medication_adherence(x_train, y_train).predict(patient_data)[0]

        if predictions[0] == 1:
            message = f"It's time to take medication_schedule[current_time]."
        else:
            message = f"It's time to take medication_schedule[current_time]. Ensure adherence"

        notification.notify(
            title="Medication Reminder",
            message=message,
            timeout=10
        )
        engine.say(message)
        engine.runAndWait()
        play_sound("voice.mp3")
        play_vibration()


# Reminder to collect medicine from hospital
def reminder_to_collect():
    current_date = datetime.now().strftime("%d-%m-%Y")
    hospital_schedule = {
        "06-09-2023": "Collect your medicine from hospital",
        "11-09-2023": "Collect your medicine from hospital",
        "25-09-2023": "Collect your medicine from hospital",
    }

    if current_date in hospital_schedule:
        message = f"Don't forget to collect your medicine today"
        notification.notify(
            title="Medicine Collection Reminder",
            message=message,
            timeout=10

        )
        engine.say(message)
        engine.runAndWait()
        play_sound("voice.mp3")
        play_vibration()


# scheduler
scheduler = BackgroundScheduler()

# medication reminders every five minutes for testing
scheduler.add_job(reminder_to_take_medication, 'interval', minutes=5)

# medication collection reminder daily 05:00 am morning
scheduler.add_job(reminder_to_collect, 'cron', hour='5', minute='0')

scheduler.start()
