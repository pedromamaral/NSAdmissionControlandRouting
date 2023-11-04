import os
import gym
from gym import Env, spaces
import numpy as np
import copy
from time import sleep
import random
import envs.networkX_api_topo as netX
from queue import Queue
from threading import Thread

# example network path selection
from envs.containernet_api_topo import ContainernetAPI
from env_examples.dynamic_network_slicing_path_selection.parameters import BASE_STATIONS, COMPUTING_STATIONS, CONNECTIONS_OFFSET,\
        INPUT_DIM, CHOICES, ELASTIC_ARRIVAL_AVERAGE, INELASTIC_ARRIVAL_AVERAGE, DURATION_AVERAGE, CONNECTIONS_AVERAGE,\
        MAX_REQUESTS, PATHS_PER_PAIR, PORT_RANGE,TOPOLOGY_FILE,REQUEST_FILE,NUMBER_PATHS,MAX_REQUESTS_QUEUE

class ContainernetEnv(Env):
    def __init__(self):
        super(ContainernetEnv, self).__init__()
        self.bottlenecks_src_dst={}
        self.containernet = ContainernetAPI(TOPOLOGY_FILE)
        
        self.graph = netX.build_graph_from_txt(self.containernet.bw_capacity,weights=True)
        
        self.upload_starting_rules()

        # define Observation Space
        low = np.zeros(INPUT_DIM, dtype=np.float32)
        high = np.array([2.0, 60.0, 100.0, 2.0] + [1.0] * BASE_STATIONS * COMPUTING_STATIONS +
                        [float(MAX_REQUESTS)] * 2 + [750.0] * NUMBER_PATHS,
                        dtype=np.float32)

        self.observation_space= spaces.Box(low=low, high=high, dtype=np.float32)

        # define Action Space
        self.action_space = spaces.Tuple((spaces.Discrete(CHOICES),spaces.Discrete(PATHS_PER_PAIR)))
        
        #define state
        self.state = np.zeros(INPUT_DIM, dtype=np.float32)

        # define Env Elements/Variables
        self.requests= 0
        self.requests_queue = Queue(maxsize=MAX_REQUESTS_QUEUE)
        self.departed_queue = Queue(maxsize=MAX_REQUESTS_QUEUE)

        self.generator_semaphore = False
        self.elastic_generator = None
        self.inelastic_generator = None
        self.evaluators = []

        self.elastic_request_templates = []
        self.inelastic_request_templates = []
        self.elastic_request_templates, self.inelastic_request_templates = read_templates(REQUEST_FILE)

        self.active_ports = []
        self.active_connections = []
        self.active_paths = BASE_STATIONS * COMPUTING_STATIONS * [-1]
        self.bottlenecks = []
        

    ''' 
        get_state function

        translates the environment state into an observation
        The agent gets to a new state or observation state is
        the information of the environment that an agent is in
        and observation is an actual image that the agent sees.
    '''

    def get_state(self):
        
        return self.state

    ''' 
        get information function
        return auxiliary information that is returned by step and reset function
    '''

    def _get_info(self):
        print("_get_info function")

    ''' 
        reset function
    '''
    # reset function

    def reset(self) -> object:
        self.containernet.clear_logs()
        self.state = np.zeros(INPUT_DIM, dtype=np.float32)

        self.requests = 0
        self.requests_queue = Queue(maxsize=MAX_REQUESTS_QUEUE)
        self.departed_queue = Queue(maxsize=MAX_REQUESTS_QUEUE)

        self.active_ports = []
        self.active_paths = BASE_STATIONS * COMPUTING_STATIONS * [-1]
        self.active_connections = []

        self.generator_semaphore = True
        self.elastic_generator = Thread(target=self.request_generator, args=(1, ))
        self.inelastic_generator = Thread(target=self.request_generator, args=(2, ))
        self.elastic_generator.start()
        self.inelastic_generator.start()
        self.evaluators = []

        self.containernet.active_paths={}
        self.containernet.ofp_match_params={}
        self.containernet.bw_available_now=copy.deepcopy(self.containernet.bw_capacity)
        self.containernet.bw_available_cumulative = copy.deepcopy(self.containernet.bw_capacity)
        self.containernet.bw_used={}

        self.state_from_request(self.requests_queue.get(block=True))

        return self.state

    # render function
    def render(self, mode="human"):
        print("render function")

    # close function
    def close(self):
        print("close function")

    # step function
    def step(self, action):
        reward: float = 0.0
        done: bool = False
        accepted_slices: int = 0
        print("ACTION IN STEP",action)
        action1,action2=action[0],int(action[1])
        
        if self.state[0]:
            self.requests += 1
            if action1:
                print(f"ACCEPT")
                accepted_slices += 1
                print("Path:",action2)
                self.create_slice(*slice_connections_from_array(self.state[4:CONNECTIONS_OFFSET]),action2)
                if self.state[0] == 1:  # elastic slice
                    self.state[CONNECTIONS_OFFSET] += 1
                elif self.state[0] == 2:  # inelastic slice
                    self.state[CONNECTIONS_OFFSET + 1] += 1
                reward = self.state[1] * self.state[3]
            else:
                print("REJECT")

        if self.requests < MAX_REQUESTS:
            self.state_from_request(self.requests_queue.get(block=True))
            if self.state[0] == 0:  # slice departure
                departure = self.departed_queue.get()
                self.state[CONNECTIONS_OFFSET + departure["type"] - 1] -= 1
                reward += departure["reward"]
        else:
            if self.generator_semaphore:
                self.stop_generators()
            for evaluator in self.evaluators:
                if evaluator.is_alive():  # might get stuck if a second evaluator finishes before this one
                    evaluator.join()
                    reward += self.state_from_departure(self.departed_queue.get())
                    # print(self.state)
                    return self.state, reward, done, {}
            while not self.departed_queue.empty():  # prevent the previous error
                reward += self.state_from_departure(self.departed_queue.get())
                # print(self.state)
                return self.state, reward, done, {}
            done = True

        # print(self.state)
        return self.state, reward, done, accepted_slices, self.requests, {}

#***************************************************************************************************************************************
#************************************************AUXILIAR FUNCTIONS*********************************************************************
#***************************************************************************************************************************************

    def upload_starting_rules(self):
        ofp_match_params={}
        paths_hops={}
        paths={}
        
        print("********************************** UPLOAD_STARTING_RULES")
        
        nodes = [node for node in self.containernet.network.hosts]
        base_stations = [bs.MAC() for bs in nodes if "B" in bs.name]
        computing_stations = [cs.MAC() for cs in nodes if "C" in cs.name or "M" in cs.name]

        for bs in base_stations:
            if bs not in self.bottlenecks_src_dst.keys():
                self.bottlenecks_src_dst[bs]={}
            for cs in computing_stations:
                    
                    path= self.get_higher_bottleneck_path(self.graph, bs, cs, PATHS_PER_PAIR)
                    paths_hops[(bs, cs)] = path
                    paths[(bs, cs)] = netX.add_ports_to_path(path, self.containernet.adjacency, bs, cs)
                    
                    if bs not in ofp_match_params:
                        ofp_match_params[bs]={}
                    ofp_match_params[bs][cs]={}
                    ofp_match_params[bs][cs]=self.containernet.define_ofp_match_params(bs, cs)


                    path = self.get_higher_bottleneck_path(self.graph, cs, bs, PATHS_PER_PAIR)
                    paths_hops[(cs, bs)] = path
                    paths[(cs, bs)] = netX.add_ports_to_path(path, self.containernet.adjacency, cs, bs)
                    
                    if cs not in ofp_match_params:     
                        ofp_match_params[cs]={}
                    ofp_match_params[cs][bs]={}
                    ofp_match_params[cs][bs]=self.containernet.define_ofp_match_params(cs, bs)


        self.containernet.upload_data_in_json_file(paths_hops, "starting_rules/paths_hops.json", "w")
        self.containernet.upload_data_in_json_file(paths, "starting_rules/paths.json", "w")
        self.containernet.upload_data_in_json_file(ofp_match_params, "starting_rules/ofp_match_params.json", "w")
        


    def create_slice(self, clients, servers,action):
        
        ports= []
        traffic_type='tcp'
        self.update_bottlenecks()
        #install rules in controller
        for (client, server) in zip(clients, servers):

            if client not in self.bottlenecks_src_dst.keys():
                self.bottlenecks_src_dst[client]={}

            client = self.containernet.get_host_mac(client)
            server = self.containernet.get_host_mac(server)
            paths = self.get_bottleneck_and_paths(self.graph, client, server, PATHS_PER_PAIR)
            if(len(paths)==1):#one path
                path=paths[0]
            else:
                path=paths[action]
            
            paths_r=self.get_bottleneck_and_paths(self.graph, server, client, PATHS_PER_PAIR)
            if(len(paths_r)==1):#one path
                path_r=paths_r[0]
            else:
                path_r=paths_r[action]
            
            self.containernet.send_path_to_controller(client,server,path,path_r)
            self.containernet.get_bw_used_bw_available()

        #create iperf traffic
        for (client, server) in zip(clients, servers):
            port: int = random.choice([port for port in range(*PORT_RANGE) if port not in self.active_ports])
            ports += [port]
            self.active_ports += [port]
            self.active_connections += [f'{client}_{server}_{port}']
            #iperf
            if self.state[0]==2:traffic_type='udp'
            self.containernet.generate_traffic_with_iperf(client, server, port, self.state[1], self.state[2],traffic_type)

        
        evaluator = Thread(target=self.slice_evaluator,
                           args=(clients, servers, ports, self.state[0], self.state[1], self.state[2], self.state[1] * self.state[3],action))
        self.evaluators += [evaluator]
        evaluator.start()

    def slice_evaluator(self, clients, servers, ports, slice_type, duration, bw, price,action) :
        
        if slice_type not in [1, 2]:
            return

        traffic_type='tcp'
        if slice_type==2:traffic_type='udp'

        sleep(duration)

        data= []
        self.update_bottlenecks()
        
        for (client, server, port) in zip(clients, servers, ports):

            if client not in self.bottlenecks_src_dst.keys():
                self.bottlenecks_src_dst[client]={}
            
            result = self.containernet.json_from_log(client, server, port, "client", traffic_type)
    
            if result:
                if "error" in result:
                    print("ERROR in connections{}_{}_{}.log".format(client,server,port))
                else:
                    data += [result]

            self.active_connections.remove(f'{client}_{server}_{port}')

            client = self.containernet.get_host_mac(client)
            server = self.containernet.get_host_mac(server)
            paths = self.get_bottleneck_and_paths(self.graph, client, server, PATHS_PER_PAIR)
            if(len(paths)==1):#one path
                path=paths[0]
            else:
                path=paths[action]
            
            paths_r=self.get_bottleneck_and_paths(self.graph, server, client, PATHS_PER_PAIR)
            if(len(paths_r)==1):#one path
                path_r=paths_r[0]
            else:
                path_r=paths_r[action]
            
            self.containernet.send_path_to_controller(client,server,path,path_r)
            self.containernet.get_bw_used_bw_available()
            
        if data:
            reward = evaluate_elastic_slice(bw, price, data) if slice_type == 1 else evaluate_inelastic_slice(bw, price, data)
        else:
            reward=0
        self.departed_queue.put(dict(type=1 if slice_type == 1 else 2, reward=reward))
        self.requests_queue.put(dict(type=0, duration=0, bw=0.0, price=0.0,
                                     connections=np.zeros(BASE_STATIONS * COMPUTING_STATIONS, dtype=np.float32)))
        


    def request_generator(self, slice_type):
        if slice_type not in [1, 2]:
            return

        while self.generator_semaphore:
            arrival: float = np.random.poisson(ELASTIC_ARRIVAL_AVERAGE if slice_type == 1 else INELASTIC_ARRIVAL_AVERAGE)
            sleep(arrival)

            if self.generator_semaphore:  # ensures req isn't created if new req is created while inside loop
                duration: int = min(max(int(np.random.exponential(DURATION_AVERAGE)), 1), 60)
                bw, price = random.choice(self.elastic_request_templates if slice_type == 1 else self.inelastic_request_templates)

                number_connections = min(max(int(np.random.exponential(CONNECTIONS_AVERAGE)), 1), BASE_STATIONS)
                base_stations = random.sample(range(BASE_STATIONS), number_connections)
                computing_stations = random.sample(range(COMPUTING_STATIONS), number_connections)

                connections = np.zeros((BASE_STATIONS, COMPUTING_STATIONS), dtype=np.float32)
                for (bs, cs) in zip(base_stations, computing_stations):
                    connections[bs][cs] = 1

                self.requests_queue.put(dict(type=slice_type, duration=int(duration), bw=float(bw),
                                             price=float(price), connections=connections.flatten()))
                


    def get_bottlenecks_list(self):
        self.bottlenecks=[]
        for src, value in self.bottlenecks_src_dst.items():
            for dst, bottleneck_list in value.items():
                for idx,bottleneck in bottleneck_list.items():
                    self.bottlenecks.append(bottleneck)
        
        return self.bottlenecks

    def state_from_request(self, request):
        self.state[0] = request["type"]
        self.state[1] = request["duration"]
        self.state[2] = request["bw"]
        self.state[3] = request["price"]
        self.state[4:CONNECTIONS_OFFSET] = request["connections"]
        self.state[CONNECTIONS_OFFSET:] = self.get_bottlenecks_list()
    
    def state_from_departure(self, departure):
        self.state[:CONNECTIONS_OFFSET] = np.zeros(4 + BASE_STATIONS * COMPUTING_STATIONS, dtype=np.float32)
        self.state[CONNECTIONS_OFFSET + departure["type"] - 1] -= 1
        self.state[CONNECTIONS_OFFSET:] = self.get_bottlenecks_list()
        return departure["reward"]

    def stop_generators(self):
        self.generator_semaphore = False
        if self.elastic_generator.is_alive():
            self.elastic_generator.join()
        if self.inelastic_generator.is_alive():
            self.inelastic_generator.join()

    def update_bottlenecks(self):
        #print("*****************************************************************************************")
        #print("********************************** UPDATE_BOTTLENECKS ***********************************")
        #print("*****************************************************************************************")
        #print(self.containernet.bw_available_now)
        for (src, dst), bw in sorted(self.containernet.bw_available_now.items()):
                self.graph[src][dst]['weight'] = min(self.containernet.bw_available_now[src, dst], self.containernet.bw_available_now[dst, src])
    
    #return the paths and bottlenecks(link with lower bandwidth)
    def get_bottleneck_and_paths(self,graph,src,dst,number_paths):
        aux=0
        paths_return=[]
        if src not in self.bottlenecks_src_dst.keys():     
            aux=1
        
        paths=netX.k_shortest_paths(graph,src,dst,number_paths)
        bottlenecks=[]
        for i,path in enumerate(paths):
            pairs=netX.convert_path_into_pairs(path)
            if pairs == []:
                bottleneck=1000
            else:
                bottleneck=min(graph[sw1][sw2]['weight'] for (sw1,sw2) in pairs)
            
            if aux==0:
                if dst not in self.bottlenecks_src_dst[src].keys():
                    self.bottlenecks_src_dst[src][dst]={}
                self.bottlenecks_src_dst[src][dst][i]=bottleneck
            bottlenecks.append(bottleneck)
        
        for path in paths:
            path.append(dst)
            path.insert(0, src)
            paths_return.append(path)
        return paths_return
    
    #return the path with higher bottleneck(link with lower bandwidth)
    def get_higher_bottleneck_path(self,graph,src,dst,number_paths):
        aux=0
        if src not in self.bottlenecks_src_dst.keys():     
            aux=1
        
        paths=netX.k_shortest_paths(graph,src,dst,number_paths)
        bottlenecks=[]
        for i,path in enumerate(paths):
            pairs=netX.convert_path_into_pairs(path)
            if pairs == []:
                bottleneck=1000
            else:
                bottleneck=min(graph[sw1][sw2]['weight'] for (sw1,sw2) in pairs)
            
            if aux==0:
                if dst not in self.bottlenecks_src_dst[src].keys():
                    self.bottlenecks_src_dst[src][dst]={}
                self.bottlenecks_src_dst[src][dst][i]=bottleneck
            bottlenecks.append(bottleneck)
        paths[bottlenecks.index(max(bottlenecks))].append(dst)
        paths[bottlenecks.index(max(bottlenecks))].insert(0, src)
        return paths[bottlenecks.index(max(bottlenecks))]


def read_templates(file):
    elastic = []
    inelastic = []
    with open(file, 'r') as request_templates:
        for template in request_templates.readlines():
            slice_type, bw, price = template.split()
            if slice_type == 'e':
                elastic += [(float(bw), float(price))]
            else:
                inelastic += [(float(bw), float(price))]
    return elastic, inelastic

def slice_connections_from_array(connections):
    parsed_connections = [connections[i:i + COMPUTING_STATIONS] for i in range(0, len(connections), COMPUTING_STATIONS)]

    clients= []
    servers= []

    for bs_idx, base_station in enumerate(parsed_connections):
        for cs_idx, connected in enumerate(base_station):
            if connected:
                clients += [f'BS{bs_idx + 1}']
                servers += [f'{"MECS" if cs_idx < COMPUTING_STATIONS // 2 else "CS"}'
                            f'{cs_idx + 1 if (cs_idx < COMPUTING_STATIONS // 2) else (cs_idx - COMPUTING_STATIONS // 2 + 1)}']
    return clients, servers

def evaluate_elastic_slice(bw, full_price, data):
    
    averages= [connection["end"]["streams"][0]["receiver"]["bits_per_second"] / 1000000.0 for connection in data]
    total_average = sum(averages) / len(averages)
    if total_average >= bw - bw * .1:
        print(f"Finished elastic slice {total_average} >= {bw}")
        return 0.0
    print(f"Failed elastic slice {total_average} < {bw}")
    return - full_price / 2

def evaluate_inelastic_slice(bw, price, data):
    worst = min(interval["streams"][0]["bits_per_second"] / 1000000.0 for connection in data for interval in connection["intervals"])
    if worst >= bw - bw * .1:
        print(f"Finished inelastic slice {worst} >= {bw}")
        return 0.0
    print(f"Failed inelastic slice {worst} < {bw}")
    return - price


