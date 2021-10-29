from flask import Flask, render_template, request, redirect
import pickle
import tweepy

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfv = pickle.load(open('tfv.pkl', 'rb'))


@app.route("/", methods=['GET','POST'])
def home():
    return render_template('index.html',y=0)


def twitter(x):
    auth = tweepy.AppAuthHandler("vlNVFVD5QerHXuyNB5Ze4s8lS", "sWaGEYdf8cjLJRjce4IEDJc6r3KjVLBsMw6lz4L2P0VELka3uo")
    list2 = []
    api = tweepy.API(auth)
    for tweet in tweepy.Cursor(api.search_tweets, q=x, include_entities=False).items(100):
        list2.append(tweet.text.split(sep=':')[-1])
    return list2


def sentiment(x):
    n = 0
    p = 0
    for i in x:
        if i == 'negative':
            n = n + 1
        else:
            p = p + 1
    return n, p


@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        pred = []
        y = 0
        x = request.form['movie']
        if len(x) != 0:
            list1 = twitter(x)
            for i in list1:
                pred.append(model.predict(tfv.transform([i])))
                n, p = sentiment(pred)
                y = p / (n + p) * 100

        else:
            y = 0

    return render_template('index.html', x=x, y=y)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True, port=8000)
