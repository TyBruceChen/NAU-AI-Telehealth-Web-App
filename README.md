# NAU-Cap-AI-Telehealth-Amazon-Web-App
This is NAU capston: AI-Telehealth, users can get prediction of COVID19 by uploading Chest X-ray image (better resolution of 224*224). Flask python is used  for building local user, FRP is used to expose it, and Amazon EC2 instance is in charge of publishing it to public Internet.

## News in this version:

A complete server with Grad-CAM prediction visualization presented at the UGrad.

## Instruction:

Install systemd:  ```sudo apt-get systemd``` <br>
Enable systemd ```file.service``` run from booting: ```systemctl enable file.service``` <br>
Stop/Start/Check status: repalce ```enable``` with ```stop/start/status``` <br>
reload the systemd file service table: ```systemctl daemon-reload``` <br>


## Debugging:

### (Amazon) Server no response:

1. Multiple port connections are built between client and server
2. Amazon Server Blackout
3. Bugs in client (check feed back from local Flask terminal), highly encourage for backup in each successully deployment stage
4. Change of public IP address by Amazon, since it's free service
5. Run incorrect frp command/configuration files
6. Figure out whether your service is HTTP or HTTPS! This is important since it can decide which http:// or https:// can be accessed.
7. Try to switch your network. Sometimes the bad connection is due to the network.

Frpc client service is built successfully but local server can not be detected (close):

![Screenshot from 2024-04-03 00-30-51](https://github.com/TyBruceChen/NAU-Cap-AI-Telehealth-Amazon-Web-App/assets/152252677/5425804b-c908-42fe-b6c7-41d683339e56)
