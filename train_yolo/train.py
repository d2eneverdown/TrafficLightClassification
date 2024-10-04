import subprocess

command = [
    'yolo',
    'task=detect',
    'mode=train',
    'data=dataset.yaml',
    'model=../yolov8n.pt',
    'epochs=50',
    'imgsz=640',
]
subprocess.run(command)