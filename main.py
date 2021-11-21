import os
import re
import nltk
import joblib
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from urllib.parse import urlparse
from urllib.parse import parse_qs
import googleapiclient.discovery
from profanity_check import predict
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']

DEVELOPER_KEY = os.environ['DEVELOPER_KEY']


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/rating', methods=["GET", "POST"])
def rating():
    try:
        youtube_url = request.form['videoURL']
        parsed_url = urlparse(youtube_url)
        video_id = parse_qs(parsed_url.query)['v'][0]

        # Extract comments data in JSON format
        def google_api(id):
            api_service_name = "youtube"
            api_version = "v3"

            youtube = googleapiclient.discovery.build(
                api_service_name, api_version, developerKey=DEVELOPER_KEY)

            request = youtube.commentThreads().list(
                part="id,snippet",
                maxResults=100,
                order="relevance",
                videoId=id
            )
            response = request.execute()

            return response

        response = google_api(video_id)

        # Create comments dataframe
        def create_df_author_comments():
            authorname = []
            comments = []
            for i in range(len(response["items"])):
                authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
                comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
            comment_df = pd.DataFrame(comments, index=authorname, columns=["Comments"])
            return comment_df

        comment_df = create_df_author_comments()

        # Clean comments
        def cleaning_comments(comment):
            comment = re.sub("[0-9]+", "", comment)
            comment = re.sub("[\:|\@|\)|\*|\.|\$|\!|\?|\,|\%|\"|\(|\-|\”|\“|\#|\!|\/|\«|\»|\&|\n|\’|\'|🇵🇰|\;|\！]+",
                             " ",
                             comment)

            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            comment = emoji_pattern.sub(r'', comment)

            return comment

        comment_df["Comments"] = comment_df["Comments"].apply(cleaning_comments)

        # Lowercase
        lower = lambda comment: comment.lower()
        comment_df['Comments'] = comment_df['Comments'].apply(lower)

        # Strip
        comment_df['Comments'] = comment_df['Comments'].str.strip()

        # Remove stopwords
        nltk.download("stopwords")
        nltk.download("punkt")
        stop_words = set(stopwords.words('english'))

        def remove_stopwords(line):
            word_tokens = word_tokenize(line)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            return " ".join(filtered_sentence)

        comment_df['Comments'] = comment_df['Comments'].apply(lambda x: remove_stopwords(x))

        # Remove empty comments
        def remove_empty_comments(df):
            zero_length_comments = df[df["Comments"].map(len) == 0]
            zero_length_comments_index = [i for i in zero_length_comments.index]
            df.drop(zero_length_comments_index, inplace=True)
            return df

        comment_df = remove_empty_comments(comment_df)

        # Load ML Model
        model_filename = 'static/assets/model/model.joblib.z'
        clf = joblib.load(model_filename)

        # Predict Ratings
        def predict_rating_of_single_comment(text):
            probas = clf.predict_proba([text.lower()])[0]
            if max(probas) == probas[0]:
                return 1
            elif max(probas) == probas[1]:
                return 2
            elif max(probas) == probas[2]:
                return 3
            elif max(probas) == probas[3]:
                return 4
            else:
                return 5

        def predict_rating_of_every_comment(df):
            df['Rating'] = df['Comments'].apply(predict_rating_of_single_comment)
            return df

        comment_df = predict_rating_of_every_comment(comment_df)

        # Calculations regarding rating
        total = comment_df['Rating'].count()
        rating_counts = comment_df['Rating'].value_counts()
        rating_keys = rating_counts.index.tolist()

        if 5 in rating_keys:
            five_rating = (rating_counts[5] / total) * 100
        else:
            five_rating = 0

        if 4 in rating_keys:
            four_rating = (rating_counts[4] / total) * 100
        else:
            four_rating = 0

        if 3 in rating_keys:
            three_rating = (rating_counts[3] / total) * 100
        else:
            three_rating = 0

        if 2 in rating_keys:
            two_rating = (rating_counts[2] / total) * 100
        else:
            two_rating = 0

        if 1 in rating_keys:
            one_rating = (rating_counts[1] / total) * 100
        else:
            one_rating = 0

        overall_rating = ((five_rating * 5) + (four_rating * 4) + (three_rating * 3) + (two_rating * 2) + (
                one_rating * 1)) / 100

        # Predict Profanity
        def predict_profanity_in_single_comment(text):
            return predict([text])[0]

        def predict_profanity_in_every_comment(df):
            df['Profanity'] = df['Comments'].apply(predict_profanity_in_single_comment)
            return df

        comment_df = predict_profanity_in_every_comment(comment_df)

        # Calculations regarding profanity
        profanity_count = comment_df['Profanity'].value_counts()[0]
        profanity_count = total - profanity_count

        overall_profanity = (profanity_count / total) * 100

        return render_template("rating.html", youtube_url=youtube_url, video_id=video_id, five_rating=five_rating, four_rating=four_rating,
                               three_rating=three_rating, two_rating=two_rating, one_rating=one_rating,
                               overall_rating=overall_rating, overall_profanity=overall_profanity)

    except:
        flash("Please enter a valid URL.", category="error")
        return redirect(url_for('home'))


if __name__ == "__main__":
    app.run()
