from flask import Flask
app = Flask(__name__)

DREAMS = ['I am going to win this event nad prove myself']

@app.route('/')
def build():
    return "In the basic route"
  
@app.route('/dreams')
def dreams():
    return DREAMS
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
