from flask import Flask,url_for,render_template,request
#from PIL import Image
from blob_storage import *
from server_model_test import *
from text_modification import *
import os,time,random

WORK_FOLDER = '/home/piko/Documents/Flask/'    #change the work folder to absolute path for command line execute 
os.chdir(WORK_FOLDER)

temp_img_path = 'temp_imgs'
#temp_img_path = 'home/site/temp_imgs'   #when it's uploaded to Azure server.
model_folder = 'models/' #blob folder for model saving
model_path = os.path.join(model_folder,'Diagnosis_Model.pt') 
model_storage_folder = model_folder
storage_path = os.path.join(model_storage_folder,'Diagnosis_Model.pt') 

sample_folder = 'imgs/'  #blob folder for sample imgs
sample_database_name_list = os.listdir('static/sample_imgs')  #the samples images contained in local should be exact same as in the blob server

app = Flask(__name__)

if os.path.exists(temp_img_path) != True:
        os.mkdir(temp_img_path) #create the local server to temporarily store the imgs

if os.path.exists(model_storage_folder) != True:
        os.mkdir(model_storage_folder) #create the local server to temporarily store the imgs

blob_server_client = blob_server_client()   #build the connection to the azure storage service
if os.path.isfile(storage_path) != True:
    blob_server_client.download_file_blob(filename = model_path, storage_name = storage_path)

try:
    server_model = load_model(model_path)   #load the model
except:
    blob_server_client.download_file_blob(filename = model_path, storage_name = storage_path) 
    server_model = load_model(model_path)   #load the model

@app.route('/temp/<var>')
def temp(var):
    # test page
    return '<!DOCTYPE html>\
            <html><head></head> \
            <body>This is a temp page</body>\
            </html>'

@app.route('/',methods = ['GET','POST'])
def file_handle():
    #wait_time = 0  #let the thread to run 10s
    if request.method == 'POST':
        print(request.files)
        if 'pic' in request.files:
            file = request.files['pic']
            try:
                upload_name = file.filename
                print(upload_name) #see the name of uploaded file
                upload_name = upload_name.split('.')[0]

                img_name = str(upload_name) + str(int(random.random()*1000)) + '.png'
                file_name = os.path.join( temp_img_path,img_name)    #temperaryly save the img on local
                file.save(file_name)
                # the initial idea is to save the file then read as binary type
                """
                with open(file_name,'rb') as bin_file:
                    blob_server_client.store_file_blob(bin_file = bin_file, filename = img_name)
                print('Upload Finish.')
                """
                try:
                    img = img_process(file_name)    #read the img, reshape, and normalize
                    y = server_model(img)   #prediction

                    try:
                        os.remove(file_name)
                        print('Server File deleted.')
                    except:
                        print('Fail to delete server file!')

                    result = int(y.argmax())
                    print(lung_type(result))
                    html_update = text_modification(replaced_element= 'waiting to be uploaded')
                    return html_update.replace_content(lung_type(result))
                except:
                    print('Fail to loaad the image')
                    print('Please don not upload empty file!')
                    html_update =  text_modification(replaced_element= 'waiting to be uploaded')
                    return html_update.replace_content('Error! Please don not upload empty file! ')

            except:
                pass
                print('Error!')           
    return render_template('/index.html', name = None)


if __name__ == '__main__':
    app.run(port = '5000')


