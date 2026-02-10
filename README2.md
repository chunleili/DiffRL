
Set env with 
```
setEnv.sh
```

Run with 
```
./run.sh
```

Render to usd with 
```
./render.sh
```

view the tensorboard
```
tensorboard --logdir=/home/chunleli/Dev/DiffRL/examples/logs/ --port=6008
```

Locally forward the port with

ssh -L 6008:localhost:6008 gnome

Locally open the browser and go to

http://localhost:6008