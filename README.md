# NAU-Cap-AI-Telehealth-Amazon-Web-App
This is NAU capston: AI-Telehealth, users can get prediction of COVID19 by uploading Chest X-ray image (better resolution of 224*224). Flask python is used  for building local user, FRP is used to expose it, and Amazon EC2 instance is in charge of publishing it to public Internet.


## Instruction:

Install systemd:  ```sudo apt-get systemd``` <br>
Enable systemd file.service run from booting: systemctl enable file.service <br>
Stop/Start/Check status: repalce ```enable``` with ```stop/start/status``` <br>
reload the systemd file service table: ```systemctl daemon-reload``` <br>
