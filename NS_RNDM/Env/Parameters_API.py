import os
#define the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
#define volume directory
volume_dir = os.path.abspath(os.path.join(working_dir,os.pardir))

DOCKER_VOLUME = os.path.join(volume_dir,'volume/')
NUMBER_SWITCHES = 20
#NUMBER_SWITCHES = 65

LOG_TIMEOUT = 90