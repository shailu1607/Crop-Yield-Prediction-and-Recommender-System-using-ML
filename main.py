import pickle
import random
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# crop recommendation
model = pickle.load(open('classifierplz.pkl', 'rb'))

# fertilizer recommendation
model2 = pickle.load(open('classifier2.pkl', 'rb'))

# yield prediction
data = pd.read_csv('cropyield.csv')

data = data.dropna()
data[' area'] = data[' area'].astype(int)
data['Production'] = data['Production'].astype(int)

encode_ferti = LabelEncoder()
data['label'] = encode_ferti.fit_transform(data['label'])
# creating the dataframe
crop = pd.DataFrame(zip(encode_ferti.classes_, encode_ferti.transform(encode_ferti.classes_)),
                    columns=['original', 'Encoded'])
crop = crop.set_index('original')

X = data.drop('Production', axis=1)
y = data['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=0)

classifier_rf = RandomForestClassifier(random_state=0)

classifier_rf.fit(X_train, y_train)

model3 = pickle.load(open('classifierplz.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def enter():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/prediction')
def pred():
    return render_template('prediction.html')


@app.route('/guide')
def guide():
    return render_template('guide.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/crppred', methods=['POST'])
def crppred():
    N1 = (request.form.get('N'))
    P1 = (request.form.get('P'))
    K1 = (request.form.get('K'))

    r = {"1": [0, 10], "2": [10, 20], "3": [20, 30], "4": [30, 40], "5": [40, 50], "6": [50, 60], "7": [60, 70],
         "8": [70, 80],
         "9": [80, 90], "10": [90, 100]}
    tempN = r[N1]
    tempP = r[P1]
    tempK = r[K1]
    # print(tempN)
    N = random.randint(tempN[0], tempN[1] + 1)
    P = random.randint(tempP[0], tempP[1] + 1)
    K = random.randint(tempK[0], tempK[1] + 1)
    # print(N)
    temperature = (request.form.get('temperature'))
    humidity = (request.form.get('humidity'))
    ph = (request.form.get('ph'))

    soil = 53
    area = (request.form.get('Area'))

    use = (request.form.get('use'))
    conc = (request.form.get('conc'))

    # print(N, P, K, temperature, humidity, ph)

    ans = model.predict(np.array([[N, P, K, temperature, humidity, ph]]))

    d = {0: 'apple', 1: 'banana', 2: 'chickpea', 3: 'coconut', 4: 'coffee', 5: 'cotton', 6: 'grapes',
         7: 'jute', 8: 'kidney beans', 9: 'lentil', 10: 'maize', 11: 'mango', 12: 'wheat', 13: 'mung bean',
         14: 'muskmelon',
         15: 'orange', 16: 'papaya',
         17: 'pigeon peas', 18: 'pomegranate', 19: 'rice', 20: 'watermelon', 21: 'wheat'}

    a = ans[0]
    resultc = d[a]
    print(resultc)

    crop = a

    result1 = model2.predict(np.array([[N, K, P]]))
    if result1[0] == 0:
        resultf = 'TEN-TWENTY SIX-TWENTY SIX'
    elif result1[0] == 1:
        resultf = 'Fourteen-Thirty Five-Fourteen'
    elif result1[0] == 2:
        resultf = 'Seventeen-Seventeen-Seventeen'
    elif result1[0] == 3:
        resultf = 'TWENTY-TWENTY'
    elif result1[0] == 4:
        resultf = 'TWENTY EIGHT-TWENTY EIGHT'
    elif result1[0] == 5:
        resultf = 'DAP'
    else:
        resultf = 'UREA'
    print(resultf)

    result_rf = classifier_rf.predict(np.array([[crop, temperature, humidity, soil, area]]))
    resulty = result_rf[0]
    print(resulty)

    if (use == "1"):
        if (conc == "1"):
            message = "You are using a high concentration of Pestidicides/Weedicides"
        elif (conc == "2"):
            message = "You are using a recommended concentration of Pestidicides/Weedicides"
        else:
            message = "You are using a recommended concentration of Pestidicides/Weedicides"
    else:
        message = "It is good not to use Pestidicides/Weedicides until it is most needed"

    return render_template('result.html', resultc=resultc, resultf=resultf, resulty=resulty, message=message)


@app.route('/disp', methods=['POST'])
def disp():
    namee = (request.form.get('name'))
    return render_template('guide.html', result=str(namee))


if __name__ == '__main__':
    app.run(debug=True)
