from flask import Flask, request
from flask_restful import Resource, Api
import json

#from flask.ext.jsonpify import jsonify
from flask import jsonify

app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)

str = []
@app.route('/all', methods=['GET'])
def api_all():
    #return jsonify(books)
    print("NINAD", jsonify(str))
    return jsonify(str)


class Company(Resource):
    def get(self, name):
        data = str
        result = "company: " + data[name]
        print("NINAD 2 ", jsonify(str))
        return jsonify(result)

api.add_resource(Company, '/Company/<name>')


if __name__ == '__main__':
    with open('financial.json') as handle:
        str = json.load(handle)


    app.run(port=5002)