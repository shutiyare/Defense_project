from flask import Flask, request, render_template, url_for, redirect, flash, session
import pickle
from markupsafe import Markup
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings
# from utils.disease import disease_dic
import io
import torch
from torchvision import transforms
from PIL import Image
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bootstrap import Bootstrap
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import bcrypt
# from utils.model import ResNet9
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode(
            'utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all()

# login_manager = LoginManager()
# login_manager.init_app(app)

# bootstrap = Bootstrap(app)
# app.config['DEBUG'] = True
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///flaskcrud.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# importing pickle files
model = pickle.load(open('models/rf_pipeline.pkl', 'rb'))
ferti = pickle.load(open('models/fertname_dict.pkl', 'rb'))
loaded_model = pickle.load(open("models/RandomForest.pkl", 'rb'))
# disease_classes = ['Apple___Apple_scab',
#                    'Apple___Black_rot',
#                    'Apple___Cedar_apple_rust',
#                    'Apple___healthy',
#                    'Blueberry___healthy',
#                    'Cherry_(including_sour)___Powdery_mildew',
#                    'Cherry_(including_sour)___healthy',
#                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                    'Corn_(maize)___Common_rust_',
#                    'Corn_(maize)___Northern_Leaf_Blight',
#                    'Corn_(maize)___healthy',
#                    'Grape___Black_rot',
#                    'Grape___Esca_(Black_Measles)',
#                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                    'Grape___healthy',
#                    'Orange___Haunglongbing_(Citrus_greening)',
#                    'Peach___Bacterial_spot',
#                    'Peach___healthy',
#                    'Pepper,_bell___Bacterial_spot',
#                    'Pepper,_bell___healthy',
#                    'Potato___Early_blight',
#                    'Potato___Late_blight',
#                    'Potato___healthy',
#                    'Raspberry___healthy',
#                    'Soybean___healthy',
#                    'Squash___Powdery_mildew',
#                    'Strawberry___Leaf_scorch',
#                    'Strawberry___healthy',
#                    'Tomato___Bacterial_spot',
#                    'Tomato___Early_blight',
#                    'Tomato___Late_blight',
#                    'Tomato___Leaf_Mold',
#                    'Tomato___Septoria_leaf_spot',
#                    'Tomato___Spider_mites Two-spotted_spider_mite',
#                    'Tomato___Target_Spot',
#                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#                    'Tomato___Tomato_mosaic_virus',
#                    'Tomato___healthy']

# disease_model_path = 'models/plant_disease_model.pth'
# disease_model = ResNet9(3, len(disease_classes))
# disease_model.load_state_dict(torch.load(
#     disease_model_path, map_location=torch.device('cpu')))
# disease_model.eval()


# def predict_image(img, model=disease_model):
#     """
#     Transforms image to tensor and predicts disease label
#     :params: image
#     :return: prediction (string)
#     """
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(io.BytesIO(img))
#     img_t = transform(image)
#     img_u = torch.unsqueeze(img_t, 0)

#     # Get predictions from model
#     yb = model(img_u)
#     # Pick index with highest probability
#     _, preds = torch.max(yb, dim=1)
#     predictionS = disease_classes[preds[0].item()]
#     # Retrieve the class label
#     return predictionS

# ad = SQLAlchemy()
# ad.init_app(app)


# class User(UserMixin):
#     def __init__(self, username, password):
#         self.id = username
#         self.password = password


# users = {
#     'user1': User('user1', 'password1'),
#     'user2': User('user2', 'password2')
# }


# class LoginForm(FlaskForm):
#     username = StringField('Username', validators=[DataRequired()])
#     password = PasswordField('Password', validators=[DataRequired()])
#     submit = SubmitField('Login')


# class SignupForm(FlaskForm):
#     username = StringField('Username', validators=[DataRequired()])
#     password = PasswordField('Password', validators=[DataRequired()])
#     submit = SubmitField('Signup')


# @login_manager.user_loader
# def load_user(user_id):
#     return users.get(user_id)


# @app.route('/')
# def home():
#     return render_template('login.html')


@app.route('/home')
def home():
    return render_template('index1.html')


@app.route('/cropdesc')
def cropdesc():
    return render_template('cropdesc.html')


@app.route('/fertdesc')
def fertdesc():
    return render_template('fertdesc.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/home')
        else:
            return render_template('logginn.html', error='Invalid user')

    return render_template('logginn.html')


# @app.route('/logout', methods=['GET', 'POST'])
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/')

    return render_template('signup.html')


# @app.route('/dashboard', methods=['GET', 'POST'])
# # login_required@
# def dashboard():
#     if session['email']:
#         user = User.query.filter_by(email=session['email']).first()
#         return render_template('dashboard.html', user=user)

#     return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')


@app.route('/predict')
def pred():
    return render_template('fert.html')


@app.route('/predictcrop')
def predicro():
    return render_template('crop.html')


@app.route('/predict', methods=['POST',])
def predict():
    temper = int(request.form.get('temper'))
    humi = int(request.form.get('humi'))
    mois = int(request.form.get('mois'))
    soil = int(request.form.get('soil'))
    crop = int(request.form.get('crop'))
    nitro = int(request.form.get('nitro'))
    pota = int(request.form.get('pota'))
    phosp = int(request.form.get('phos'))

    nitro_error = ''
    if nitro < 4:
        nitro_error = 'Nitrogen ARE NOT LESS THAN 4.'

    elif nitro > 42:
        nitro_error = 'Nitrogen value is too high.'

    temper_error = ''
    if temper < 25:
        temper_error = 'temper value  must be greater than 25.'

    elif temper > 38:
        temper_error = 'temper value is too high.'

    humi_error = ''
    if humi < 50:
        humi_error = 'humidity value must be greater than 50.'

    elif humi > 72:
        humi_error = 'humidity value is too high.'

    mois_error = ''
    if mois < 25:
        mois_error = 'moisture value  must be greater than 25.'

    elif mois > 65:
        mois_error = 'moisture value is too high.'

    pota_error = ''
    if pota < 0:
        pota_error = 'photosouim value  must be greater than 1.'

    elif pota > 19:
        pota_error = 'Potassium value is too high.'

    phos_error = ''
    if phosp < 0:
        phos_error = 'Phosporus value must be greater than 1.'

    elif phosp > 42:
        phos_error = 'Phosporus value is too high.'

    if nitro_error or phos_error or pota_error or temper_error or humi_error or mois_error:
        return render_template('fert.html', nitro_error=nitro_error, phos_error=phos_error, pota_error=pota_error,
                               temper_error=temper_error, humi_error=humi_error, mois_error=mois_error)

    input = [temper, humi, mois, soil, crop, nitro, pota, phosp]

    res = ferti[model.predict([input])[0]]

    return render_template('fert.html', x=('Predicted Fertilizer is {}'.format(res)))


# @app.route('/predictcrop', methods=['POST'])
# def predictcrop():
#     N = int(request.form['Nitrogen'])
#     P = int(request.form['Phosporus'])
#     K = int(request.form['Potassium'])
#     temp = float(request.form['Temperature'])
#     humidity = float(request.form['Humidity'])
#     ph = float(request.form['pH'])
#     rainfall = float(request.form['Rainfall'])


#     feature_list = [N, P, K, temp, humidity, ph, rainfall]
#     single_pred = np.array(feature_list).reshape(1, -1)

#     prediction = loaded_model.predict(single_pred)

#     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                  8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                  14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                  19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there".format(
#             crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

#     return render_template('home.html', prediction=result)
@app.route('/predictcrop', methods=['POST'])
def predictcrop():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    nitrogen_error = ''
    if N < 0:
        nitrogen_error = 'Nitrogen value must be non-negative.'

    elif N > 140:
        nitrogen_error = 'Nitrogen value is too high.'

    # if nitrogen_error:
    #     return render_template('home.html', nitrogen_error=nitrogen_error)
    phosphorus_error = ''
    if P < 5:
        phosphorus_error = 'Phosphorus value must be non-negative.'

    elif P > 145:
        nitrogen_error = 'Nitrogen value is too high.'

    potassium_error = ''
    if K < 5:
        potassium_error = 'Potassium value must be non-negative.'

    elif K > 205:
        potassium_error = 'Potassium value is too high .'

    temperature_error = ''
    if temp < 8.825675:
        temperature_error = 'Temperature must be lesson than  8.825675.'

    elif temp > 43.67549:
        temperature_error = 'Temperature must be greater than 43.67549.'

    humidity_error = ''
    if humidity < 14.25804 and humidity > 99.98188:
        humidity_error = 'Humidity must be between 0 and 100 %.'

    ph_error = ''
    if ph < 3.504752 and ph > 9.935091:
        ph_error = 'pH value must be between 3.504752 and 9.935091.'

    rainfall_error = ''
    if rainfall > 20.21127 and rainfall < 298.5601:
        rainfall_error = 'Rainfall value must be between 20 and 298.'

    # Check if there are any input errors
    if nitrogen_error or phosphorus_error or potassium_error or temperature_error or humidity_error or ph_error or rainfall_error:
        return render_template('crop.html', nitrogen_error=nitrogen_error, phosphorus_error=phosphorus_error, potassium_error=potassium_error, temperature_error=temperature_error, humidity_error=humidity_error, ph_error=ph_error, rainfall_error=rainfall_error)

    # If there are no input errors, perform the prediction and display the result

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(
            crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('crop.html', prediction=result)


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)

    return redirect('/')


@app.route('/disease_pred')
def predco():
    return render_template('disease.html')


# @ app.route('/disease')
# def disease_prediction():
#     title = 'Harvestify - disease Suggestion'
#     return render_template('disease.html', title=title)
# render disease prediction input page


# @app.route('/disease_pred', methods=['POST'])
# def disease_pred():
#     title = 'Harvestify - Disease Detection'
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return render_template('disease.html', title=title)
#         try:
#             img = file.read()
#             predictionS = predict_image(img)
#             predictionS = Markup(str(disease_dic[predictionS]))
#             return render_template('disease-result.html', predictionS=predictionS, title=title)
#         except:
#             pass
#     return render_template('disease.html', title=title)


if __name__ == "__main__":
    app.run(debug=True)

# from app.module.controller import *
