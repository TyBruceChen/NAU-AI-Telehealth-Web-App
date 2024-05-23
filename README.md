# NAU-Cap-AI-Telehealth-Web-App
This is an applied artificial intelligence project under the supervision of [the Digital Systems Design Laboratory of NAU](https://www.dsdlab.nau.edu/): [AI-Telehealth](https://sites.google.com/nau.edu/ai-telehealth/home), users can get the diagnosis of COVID-19 by uploading a chest X-ray image (better resolution of 224*224). Flask python is used  for building local users, [frp](https://github.com/fatedier/frp) is used to expose it, and Amazon EC2 instance is in charge of publishing it to the public Internet.
This online realization is based on the product of our research: [A Vision Transformer Machine Learning Model for COVID-19 Dagnosis Using Chest X-Ray Images](https://github.com/TyBruceChen/Research-A-Vision-Transformer-Machine-Learning-Model-for-COVID-19-Dagnosis-Using-Chest-X-Ray-Images).

## News in this version:

For this stage of the web app, content has been finalized. (Without crypto part)

## Instruction:

**Run the server from the script:**
python app.py

**Run once the computer starts:**
Install systemd:  ```sudo apt-get systemd``` <br>
Enable systemd ```file.service``` run from booting: ```systemctl enable file.service``` <br>
Stop/Start/Check status: repalce ```enable``` with ```stop/start/status``` <br>
reload the systemd file service table: ```systemctl daemon-reload``` <be>

## How the web app actually looks like:
![AI Telehealth Flask Web App1](https://github.com/TyBruceChen/NAU-AI-Telehealth-Web-App/assets/152252677/5a429904-2cc3-40c6-8c9f-5e9ec274560d)


## Debugging:

### (Amazon) Server no response:

1. Multiple port connections are built between client and server
2. Amazon Server Blackout
3. Bugs in the client (check feedback from local Flask terminal), highly encourage for backup in each successfully deployment stage
4. Unaware of changing of public IP address by Amazon, since it's a free service
5. Run incorrect frp command/configuration files
6. Figure out whether your service is HTTP or HTTPS! This is important since it can decide which http:// or https:// can be accessed.
7. Try to switch your network. Sometimes the bad connection is due to the network.

Frpc client service is built successfully but the local server can not be detected (close):

![Screenshot from 2024-04-03 00-30-51](https://github.com/TyBruceChen/NAU-Cap-AI-Telehealth-Amazon-Web-App/assets/152252677/5425804b-c908-42fe-b6c7-41d683339e56)
