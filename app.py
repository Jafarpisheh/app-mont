from flask import Flask, render_template, request, jsonify
from chatbot import ChatBot
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mont.db'
db = SQLAlchemy(app)
chatbot = ChatBot()


def msg_entry(**kwarg):
    new_msg = message(**kwarg)
    db.session.add(new_msg)
    db.session.commit()

class message(db.Model):
    message_id = db.Column(db.Integer, primary_key = True) 
    # session_id = db.Column(db.Integer, db.ForeignKey('session_id')) 
    # user_id = db.Column(db.Integer, db.ForeignKey('user_id')) 
    sender = db.Column(db.String, nullable = False) 
    content = db.Column(db.String, nullable = False) 
    created_at = db.Column(db.DateTime, default = datetime.now())  

    def __repr__(self):
        return '<message %r>' % self.message_id
    

@app.route('/',methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/get-response', methods = ['POST'])
def get_response_endpoint ():
    user_message = request.json.get('message')  
    msg_entry(content = user_message, sender = 'user')
    response = chatbot.get_response(user_message)  # Call your chatbot's function
    msg_entry(content = response, sender = 'system')
    return jsonify({'response': response})




if __name__ == "__main__":
    app.run(debug = True)