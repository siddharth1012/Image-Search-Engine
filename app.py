import pandas as pd
from flask import Flask, render_template, request
import pickle
import pandas
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_nltk = pickle.load(open('data_nltk.pkl', 'rb'))
data = pickle.load(open('image_caption_data.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['post'])
def recommend():
    text1 = request.form.get('user_input')
    text1 = text1.lower()

    ps = PorterStemmer()

    def stem(txt):
        li = []
        for i in txt.split():
            li.append(ps.stem(i))
        return " ".join(li)

    text1 = stem(text1)
    text1 = pd.DataFrame({'caption': [text1]})
    image_data = data_nltk.append(text1, ignore_index=True).copy()

    cv = CountVectorizer(max_features=100000, stop_words='english')

    vectors = cv.fit_transform(image_data['caption']).toarray()
    last_vect = vectors[len(image_data) - 1].reshape(1, -1)

    similarity = cosine_similarity(last_vect, vectors)
    distances = similarity[0]
    image_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]

    list1 = []
    for i in image_list:
        list1.append(data.iloc[i[0]].image)

    image_data.drop(image_data.tail(1).index, inplace=True)

    return render_template('index.html', list1=list1)


if __name__ == "__main__":
    app.run(debug=True)
