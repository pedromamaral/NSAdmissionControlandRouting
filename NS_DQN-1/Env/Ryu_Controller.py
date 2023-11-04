from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib.packet import packet, arp, ethernet
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link

from collections import defaultdict
from operator import attrgetter
import time
import json
import os

import NetworkX_API_Topo

from Parameters_API import DOCKER_VOLUME,NUMBER_SWITCHES

"""RYU Documentation - https://osrg.github.io/ryu-book/en/html/index.html"""



# from mininet
""" A nested dictionary that maps a host's MAC address and a
    switch's Id to the switch port that connects them.
    dict[[host's MAC address][switch]]=switch_port_to_host"""
host_to_switch_port = {}
"""host_mac{ip:ip,sw_id:port_out}"""
host_mac_ip_sw_port_out = {}

"""{"('S1', '00:00:00:00:00:01')": 100, "('00:00:00:00:00:02', 'S2')": 100}"""
bw_capacity={}
"""dict[(src_mac, dst_mac)]=[[src_mac,sw1,dst_mac],...]"""
active_paths = {}
"""dict[(sw1,sw2)]=bandwidth"""
bw = {}

# from ryu
""" Holds pairs of connected switches as keys and the port through which
    packets must leave the first switch to reach the second as value.
    dict[(switch1,switch2),switch1_port_out]"""
adjacency = {}

"""dict[(src_mac, dst_mac)]=[("switch", "in_port", "out_port"),...]"""
paths = {}

"""dict[(datapathid,sw)]=stat.tx_bytes"""
byte = {}
"""dict[(datapathid,sw)]=time.time()"""
clock = {}
"""dict[(node1,node2)]=bandwidth used"""
bw_used = {}
"""dict[(node1,node2)]=bandwidth available"""
bw_available = {}

"""[switch_id]"""
_switches = []
"""dict[(src_mac, dst_mac)]=path calculated by dijkstra_from_macs"""
paths_hops = {}

"""
    get json with key tuples and conver it into a dict
"""
def convert_json_with_key_tuples_into_dict(data):
    data_dict={}
    for devices in data:
        devices_aux = devices.replace("(", "")
        devices_aux = devices_aux.replace(")", "")
        devices_aux = devices_aux.replace(" ", "")
        devices_aux = devices_aux.replace("'", "")
        device1,device2 = devices_aux.split(",")
        data_dict[(device1,device2)]=data[devices]
    return data_dict

"""
    get data and save it in a json file given a expecific name and open_model
"""

def upload_data_in_json_file(data, file_name, open_model):

    filename = DOCKER_VOLUME+file_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if open_model != "r" or open_model != "r+":
        with open(filename, open_model) as f:
            json.dump({str(k): v for k, v in data.items()}, f)
            f.close()


def load_paths(): 
    global active_paths
    active_paths.clear()
    active_paths_json=get_data_from_json("active_paths.json")
    active_paths=convert_json_with_key_tuples_into_dict(active_paths_json)
    

def get_data_from_json(file_name):
    data = {}
    filename=DOCKER_VOLUME+file_name
    while data=={}:
        try:
            with open(filename, 'r') as f:
                if os.stat(filename).st_size == 0:
                    pass
                else:
                    data = json.loads(f.read())
            f.close()
        except:
            pass

    return data

def upload_bw_links():
    global bw,_switches,adjacency,bw_capacity

    bw_json = get_data_from_json("topology/bandwidth_links.json")
    bw_capacity=convert_json_with_key_tuples_into_dict(bw_json)
    
    for (device1,device2) in bw_capacity:
        if "S" in device1 and "S" in device2:
            device1_bw = device1.replace("S", "")
            device2_bw = device2.replace("S", "")
            bw[(device1_bw, device2_bw)] = bw_capacity[(device1, device2)]
            _switches.append(device1_bw) if device1_bw not in _switches else None
            _switches.append(device2_bw) if device2_bw not in _switches else None
    
    adjacency_json= get_data_from_json("topology/switches_adjacency.json")
    adjacency=convert_json_with_key_tuples_into_dict(adjacency_json)
    
    

def upload_topology_information():
    upload_bw_links()


class RyuController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RyuController, self).__init__(*args, **kwargs)
        self.monitor_thread = hub.spawn(self._monitor)
        

        self.topology_api_app = self

        """
            ->The SDN Datapath is a logical network device that exposes visibility and
            uncontested control over its advertised forwarding and data processing capabilities
            ->The datapath ID of an OpenFlow instance contains the instance ID and the upper
            48 bits are the bridge MAC address of the device
            -> Datapath is a class to describe an OpenFlow switch connected to this controller"""
        self.datapaths = {}

    """
        In order to detect errors and identify causes, it is necessary to understand the network status on a regular basis.
            Furthermore, a traffic monitoring system is assembled using two methods, _monitor
        and _request_stats. Every five seconds, they send an OFPPortStatsRequest message to each
        datapath registered in the controller.
             constant monitoring of the health of a network is essential for continuous and safe operation of the services
        or businesses that use that network. As a matter of course, simply monitoring traffic information does not provide
        a perfect guarantee but this section describes how to use OpenFlow to acquire statistical information for a switch.
        =>In thread function _monitor(), issuance of a statistical information acquisition request for the registered
        switch is repeated infinitely every 10 seconds"""

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(3)

    """
        With periodically called _request_stats(), OFPFlowStatsRequest and OFPPortStatsRequest are issued to the switch."""

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        """ =>OFPFlowStatsRequest requests that the switch provide statistical information related to flow entry.
        The requested target flow entry can be narrowed down by conditions such as table ID, output port,
        cookie value and match but here all entries are made subject to the request."""
        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)

        """=>OFPPortStatsRequest request that the switch provide port-related statistical information.
        It is possible to specify the desired port number to acquire information from.
        Here, OFPP_ANY is specified to request information from all ports."""
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    """
        =>In order to make sure the connected switch is monitored,
    EventOFPStateChange event is used for detecting connection and disconnection.
    This event is issued by the Ryu framework and is issued when the Datapath state is changed.
    Here, when the Datapath state becomes MAIN_DISPATCHER, that switch is registered as the monitor
    target and when it becomes DEAD_DISPATCHER, the registration is deleted.
         OpenFlow switch states
            HANDSHAKE_DISPATCHER - Exchange of HELLO message
            CONFIG_DISPATCHER - Waiting to receive SwitchFeatures message
            MAIN_DISPATCHER - Normal status
            DEAD_DISPATCHER -  Disconnection of connection"""
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        #print("EventOFPStateChange")

        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                print('Datapath {} registered.'.format(datapath.id))
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                print('Datapath {} unregistered.'.format(datapath.id))
                del self.datapaths[datapath.id]

        upload_topology_information()
        
        if len(self.datapaths) == NUMBER_SWITCHES:
            
            print("upload topo and install starting rules")
            self.install_starting_rules()




    def install_starting_rules(self):
        global paths, paths_hops
        
        print("*****************************************************************************************")
        print("******************************* INSTALL_STARTING RULES **********************************")
        print("*****************************************************************************************")

        paths_hops_json=get_data_from_json("starting_rules/paths_hops.json")
        paths_hops=convert_json_with_key_tuples_into_dict(paths_hops_json)
        paths_json=get_data_from_json("starting_rules/paths.json")
        paths=convert_json_with_key_tuples_into_dict(paths_json)
        ofp_match_params= get_data_from_json("starting_rules/ofp_match_params.json")

        for (src_host,dst_host) in paths:
            self.install_path(paths[(src_host, dst_host)], ofp_match_params[src_host][dst_host])

    """
        =>After handshake with the OpenFlow switch is completed, the Table-miss flow entry is
        added to the flow table to get ready to receive the Packet-In message.
        =>Specifically, upon receiving the Switch Features(Features Reply) message,
        the Table-miss flow entry is added.

        Features request message
        The controller sends a feature request to the switch upon session establishment.
        This message is handled by the Ryu framework, so the Ryu application do not need to process this typically.

            Create a table-miss entries - The table-miss flow entry specifies how to process packets that were not matched by
        other flow entries in the flow table. The table-miss flow entry wildcards all match fields (all fields omitted)
        and has the lowest priority 0.
        The table-miss flow entry behaves in most ways like any other flow entry.
    """
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

       # print("EventOFPSwitchFeatures")
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()

        """
            =>The Table-miss flow entry has the lowest (0) priority and this entry matches all packets.
            In the instruction of this entry, by specifying the output action to output to the controller port,
            in case the received packet does not match any of the normal flow entries, Packet-In is issued.
                An empty match is generated to match all packets. Match is expressed in the OFPMatch class.
                Next, an instance of the OUTPUT action class (OFPActionOutput) is generated to transfer
            to the controller port. The controller is specified as the output destination and OFPCML_NO_BUFFER is specified
            to max_len in order to send all packets to the controller.
                Finally, 0 (lowest) is specified for priority and the add_flow() method is executed to send the Flow Mod message.
                The content of the add_flow() method is explained in a later section.
        """
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]

        self.add_flow(datapath, 0, match, actions)

    """
        For flow entries, set match that indicates the target packet conditions, and instruction that
    indicates the operation on the packet, entry priority level, and effective time.
        Apply Actions is used for the instruction to set so that the specified action is immediately used.
        Finally, add an entry to the flow table by issuing the Flow Mod message.
        The class corresponding to the Flow Mod message is the OFPFlowMod class.
        The instance of the OFPFlowMod class is generated and the message is sent to
    the OpenFlow switch using the Datapath.send_msg() method.
    """

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)

        datapath.send_msg(mod)

    def remove_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(datapath=datapath, match=match, priority=1,
                                command=ofproto.OFPFC_DELETE, out_group=ofproto.OFPG_ANY, out_port=ofproto.OFPP_ANY)

        datapath.send_msg(mod)

    """
            The switches that receive the stats request from the monitoring system will return a
        message with a stats object. For each stat received whose port matches a port saved in the
        adjacency dictionary, the switch pair link's available bandwidth is calculated, as well as
        its cost and number of flows. Due to changes in the links' costs, the paths between host
        pairs must be updated. Once the correct paths are calculated, the existing flow rules are
        uninstalled from the switches and replaced with the most recent paths.

        =>OPFPortStatsReply class's attribute body is the list of OFPPortStats.
        =>OFPPortStats stores statistical information such as port numbers, send/receive packet count,
        respectively, byte count, drop count, error count, frame error count, overrun count, CRC error count,
        and collision count.
        =>Here, being sorted by port number, receive packet count, receive byte count, receive error count,
        send packet count, send byte count, and send error count are output."""
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available, _switches, bw

        statistics = {}
        bw_values = {}

        body = ev.msg.body
        dpid = ev.msg.datapath.id
        statistics[dpid] = {}
        bw_values[dpid] = {}
        
        for stat in sorted(body, key=attrgetter('port_no')):
            for sw in _switches:
            
                if adjacency.get((str(dpid), sw), 0) == stat.port_no:
                    
                    if int(byte.get((str(dpid), sw), 0)) > 0 and clock.get((str(dpid), sw)):

                        bw_used[(str(dpid), sw)] = round((stat.tx_bytes - float(byte.get((str(dpid), sw), 0))) * 8.0 *(10**-6)\
                            / (time.time() - float(clock.get((str(dpid), sw), 0))),0)
                        
                        bw_available[(str(dpid), sw)] = int(bw.get((str(dpid), sw), 0)) - float(bw_used.get((str(dpid), sw), 0))

                        print(bw_used[(str(dpid), sw)] ,bw_available[(str(dpid), sw)])
                        
                    byte[(str(dpid), sw)] = stat.tx_bytes
                    clock[(str(dpid), sw)] = time.time()

                    statistics[dpid][stat.port_no] = {}
                    statistics[dpid][stat.port_no]['dst_sw_id'] = sw
                    statistics[dpid][stat.port_no]['collisions'] = stat.collisions
                    statistics[dpid][stat.port_no]['duration_nsec'] = stat.duration_nsec
                    statistics[dpid][stat.port_no]['duration_sec'] = stat.duration_sec
                    statistics[dpid][stat.port_no]['rx_bytes'] = stat.rx_bytes
                    statistics[dpid][stat.port_no]['rx_crc_err'] = stat.rx_crc_err
                    statistics[dpid][stat.port_no]['rx_dropped'] = stat.rx_dropped
                    statistics[dpid][stat.port_no]['rx_errors'] = stat.rx_errors
                    statistics[dpid][stat.port_no]['rx_frame_err'] = stat.rx_frame_err
                    statistics[dpid][stat.port_no]['rx_over_err'] = stat.rx_over_err
                    statistics[dpid][stat.port_no]['rx_packets'] = stat.rx_packets
                    statistics[dpid][stat.port_no]['tx_bytes'] = stat.tx_bytes
                    statistics[dpid][stat.port_no]['tx_dropped'] = stat.tx_dropped
                    statistics[dpid][stat.port_no]['tx_errors'] = stat.tx_errors
                    statistics[dpid][stat.port_no]['tx_packets'] = stat.tx_packets

                    bw_values[dpid][sw] = {}
                    bw_values[dpid][sw]["bandwidth"] = bw.get(
                        (str(dpid), sw), 0)
                    bw_values[dpid][sw]["bandwidth_used"] = bw_used.get(
                        (str(dpid), sw), 0)
                    bw_values[dpid][sw]["bandwidth_available"] = bw_available.get(
                        (str(dpid), sw), 0)

        upload_data_in_json_file(
            statistics, "switches/sw"+str(dpid)+"/statistics_ports_sw"+str(dpid)+".json", "w")
        upload_data_in_json_file(
            bw_values, "switches/sw"+str(dpid)+"/bandwidth_sw"+str(dpid)+".json", "w")
        
        
        if len(self.datapaths) == NUMBER_SWITCHES:
            
            load_paths()
            self.update_paths()


    def update_paths(self):
        global paths_hops, paths
        params=get_data_from_json("OFPMatch/OFPMatch_params.json")
        for path in active_paths.values():
            src_mac =params[path[0]][path[-1]]["eth_src"]
            dst_mac = params[path[0]][path[-1]]["eth_dst"]
            ofpmatch_params=params[path[0]][path[-1]]
           
            saved_path = paths_hops.get((src_mac, dst_mac))
            if saved_path != path:
                self.uninstall_path(paths[(src_mac, dst_mac)],ofpmatch_params)
                paths_hops[(src_mac, dst_mac)] = path
                paths[(src_mac, dst_mac)] = NetworkX_API_Topo.add_ports_to_path(path, adjacency, src_mac, dst_mac)
                self.install_path(paths[(src_mac, dst_mac)], ofpmatch_params)

    # OFPMatch source: https://ryu.readthedocs.io/en/latest/ofproto_v1_3_ref.html#flow-match-structure
    def install_path(self, p,params):
       
        for sw, in_port, out_port in p:
            datapath = self.datapaths.get(int(sw))
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=in_port, **params)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 1, match, actions)
            
            
    def uninstall_path(self, p, params):
        
        for sw, in_port, _ in p:
            datapath = self.datapaths.get(int(sw))
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=in_port, **params)
            self.remove_flow(datapath, match)

    """ 
        =>In order to receive a response from the switch, create an event handler that receives the FlowStatsReply message.
        OPFFlowStatsReply class’s attribute body is the list of OFPFlowStats and stores the statistical information of each flow entry,
        which was subject to FlowStatsRequest.
        =>All flow entries are selected except the Table-miss flow of priority 0. The number of packets and bytes matched to
        the respective flow entry are output by being sorted by the received port and destination MAC address.
            source:https://osrg.github.io/ryu-book/en/html/traffic_monitor.html
        """
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        
        #print("EventOFPFlowStatsReply")

        dpid = ev.msg.datapath.id                       
        
        upload_data_in_json_file(ev.msg.to_jsondict(),"switches/sw"+str(dpid)+"/statistics_flow_sw"+str(dpid)+".json","w")
    
    """
        output example if -observe-links option
        [1, 2, 3, 4]
        [(2, 3, {'port': 3}),
         (3, 2, {'port': 2}),
          (3, 4, {'port': 3}),
           (2, 1, {'port': 2}), (4, 3, {'port': 2}), (1, 2, {'port': 2})]

        ->get_switch() to get the list of objects Switch
        ->get_link() to get the list of objects Link
        ->port_out : Notice that we also get the port from the source node
        that arrives at the destination node, as that information will be 
        necessary later during the forwarding step.
    
    does not get all the links
    """
    def get_ryu_switches_links(self):

        print("EventSwitchEnter")
        print("SAVE DATA ABOUT SWITCHS")
        global adjacency
        switch_list = get_switch(self.topology_api_app, None)
        for switch in switch_list:
            print("sw id")
            print(str(switch.dp.id))
        links_list = get_link(self.topology_api_app, None)
        # links=[(link.src.dpid,link.dst.dpid,{'port_out':link.src.port_no}) for link in links_list]
        print(links_list)
        for link in links_list:
            print("sw id with link")
            print(link.src.dpid,link.dst.dpid)
            adjacency[(str(link.src.dpid),str(link.dst.dpid))]=link.src.port_no
          