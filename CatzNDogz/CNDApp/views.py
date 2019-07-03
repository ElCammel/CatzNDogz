from flask import Flask, request, render_template

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')

pet_name = 'bob'
# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        return render_template('result.html', pet=pet_name)


if __name__ == "__main__":
    app.run()
