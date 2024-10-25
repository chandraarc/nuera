
import pymongo

def handle_uploaded_file(file):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    collection = myclient.sample_collection
    collection.insert_one(file)


handle_uploaded_file()