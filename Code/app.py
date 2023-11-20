from flask import Flask, render_template, request
import test_utils

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    tweet = None
    target = None
    color = "black"

    if request.method == 'POST':
        # Get values from the form
        tweet = request.form['tweet']
        target = request.form['target']
        model = request.form["model"]

        result, color = test_utils.process_web_input(tweet, target, model)

    return render_template('index.html', result=result, input_tweet=tweet or "", input_target=target, color=color)

if __name__ == '__main__':
    app.run(debug=True)
