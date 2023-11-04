import os
#define the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
#define volume directory
volume_dir = os.path.abspath(os.path.join(working_dir,os.pardir,os.pardir))

TOPOLOGY_FILE = os.path.join(working_dir,'topology.txt')
DOCKER_VOLUME = os.path.join(volume_dir,'volume/')
REQUEST_FILE= os.path.join(working_dir,'request_templates.txt')
NUMBER_SWITCHES = 64
BASE_STATIONS: int = 7
COMPUTING_STATIONS: int = 14
PATHS_PER_PAIR: int = 7
PORT_RANGE = (1024, 4097)

CONNECTIONS_OFFSET = 4 + BASE_STATIONS * COMPUTING_STATIONS
NUMBER_PATHS=BASE_STATIONS * COMPUTING_STATIONS * PATHS_PER_PAIR -44

ELASTIC_ARRIVAL_AVERAGE = 4
INELASTIC_ARRIVAL_AVERAGE = 8
DURATION_AVERAGE = 15
CONNECTIONS_AVERAGE = 2

MAX_REQUESTS_QUEUE=1000
MAX_REQUESTS= 50
STARTUP_TIME = 20
LOG_TIMEOUT = 90

INPUT_DIM = 6 + BASE_STATIONS * COMPUTING_STATIONS * (1 + PATHS_PER_PAIR) -44 # 44=7*14*7-7*6 bs-mecs so tem um caminho porque so tem um switch ligado
HL1: int = 1800
HL2: int = 1200
HL3: int = 800
CHOICES: int = 2
OUTPUT_DIM1:int=CHOICES
OUTPUT_DIM2:int=PATHS_PER_PAIR

GAMMA: float = 0.9
EPSILON: float = 0.3
LEARNING_RATE: float = 1e-3

EPOCHS: int = 5000
MEM_SIZE: int = 1000
BATCH_SIZE: int = 200
SYNC_FREQ: int = 500
