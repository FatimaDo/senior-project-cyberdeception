import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word\_tokenize
import logging
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import xml.etree.ElementTree as ET
import subprocess

# Download NLTK data

nltk.download('punkt')
nltk.download('stopwords')

# Set up logging

def setup\_logging():
logger = logging.getLogger("TangleNet")
logger.setLevel(logging.DEBUG)

```
file_handler = logging.FileHandler("TangleNet.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

return logger
```

logger = setup\_logging()

# Define the attack phases as states

attack\_phases = \[
'Reconnaissance',
'Weaponization',
'Delivery',
'Exploitation',
'Installation',
'Command and Control',
'Action'
]

# Actions that the system can take

actions = \['Allow Command', 'Substitute Command', 'Block Command']

# Rewards for each state-action pair

rewards = np.array(\[
\[10, 5, 0],   # Reconnaissance phase
\[20, 10, 0],  # Weaponization phase
\[30, 15, 0],  # Delivery phase
\[40, 20, 0],  # Exploitation phase
\[50, 25, 0],  # Installation phase
\[60, 30, 0],  # Command and Control phase
\[100, 50, 0]  # Action phase (final phase)
])

# Initialize Q-table with zeros

num\_states = len(attack\_phases)
num\_actions = len(actions)
Q = np.zeros((num\_states, num\_actions))

class AgentOne:
def **init**(self, phase\_commands):
self.phase\_commands = phase\_commands

```
def calculate_similarity(self, command1, command2):
    stop_words = set(stopwords.words('english'))
    tokens1 = [token.lower() for token in word_tokenize(command1) if token.isalnum()]
    tokens2 = [token.lower() for token in word_tokenize(command2) if token.isalnum()]
    filtered_tokens1 = [token for token in tokens1 if token not in stop_words]
    filtered_tokens2 = [token for token in tokens2 if token not in stop_words]
    intersection = len(set(filtered_tokens1).intersection(filtered_tokens2))
    union = len(set(filtered_tokens1).union(filtered_tokens2))
    similarity_score = intersection / union if union != 0 else 0.0
    return similarity_score

def identify_phase(self, command):
    best_similarity = 0
    best_phase = attack_phases[0]  # Default to the first phase
    for phase, cmds in self.phase_commands.items():
        for stored_command in cmds:
            similarity = self.calculate_similarity(command, stored_command)
            if similarity > best_similarity:
                best_similarity = similarity
                best_phase = phase
    return best_phase
```

class AgentTwo:
def **init**(self, attack\_phases, actions, rewards, phase\_commands):
self.attack\_phases = attack\_phases
self.actions = actions
self.rewards = rewards
self.num\_states = len(attack\_phases)
self.num\_actions = len(actions)
self.Q = np.zeros((self.num\_states, self.num\_actions))
self.current\_phase\_index = 0  # Track the current phase
self.phase\_commands = phase\_commands  # Save commands for each phase

```
    # Define Q-learning hyperparameters here
    self.learning_rate = 0.8
    self.discount_factor = 0.95
    self.num_episodes = 1000

def q_learning_process(self):
    # Execute commands in the current phase automatically
    while self.current_phase_index < self.num_states:  # Continue until reaching the final phase
        # Get commands for the current phase
        commands = self.phase_commands[self.attack_phases[self.current_phase_index]]
        for command in commands:
            response = execute_command(command)
            print(f"Command executed: {command}")
            print(f"Response: {response}")
            logger.info(f"Command: {command}, Response: {response}")

            # Static reward based on phase
            reward = self.rewards[self.current_phase_index, 0]  # Fetch reward from table

            # Select an action based on the current phase
            action =0  # Choose an action randomly

            # Display the selected phase and action
            print(f"Phase: {self.attack_phases[self.current_phase_index]}, Action: {self.actions[action]}")
            logger.info(f"Phase: {self.attack_phases[self.current_phase_index]}, Action: {self.actions[action]}")

            # Update Q-values based on the action taken
            old_q_value = self.Q[self.current_phase_index, action]
            self.Q[self.current_phase_index, action] = (
                (1 - self.learning_rate) * old_q_value
                + self.learning_rate * (reward + self.discount_factor * np.max(self.Q[self.current_phase_index]))
            )

            print(f"Old Q-value: {old_q_value}, Updated Q-value: {self.Q[self.current_phase_index, action]}")

        self.current_phase_index += 1  # Move to the next phase

    print("All phases executed. Final flag set!")
    print("Learned Q-values:")
    print(self.Q)
```

# Function to execute command on the server

def execute\_command(command):
try:
result = subprocess.check\_output(command, shell=True, stderr=subprocess.STDOUT)
return result.decode().strip()  # Return decoded output
except subprocess.CalledProcessError as e:
return f"Error executing command: {e.output.decode().strip()}"

# Function to read commands from the XML file

def read\_commands\_from\_file(file\_path):
phase\_commands = {phase: \[] for phase in attack\_phases}
tree = ET.parse(file\_path)
root = tree.getroot()
for phase in root.findall('Phase'):
phase\_name = phase.get('name')
commands = phase.findall('.//Code')
if phase\_name in phase\_commands:
phase\_commands\[phase\_name].extend(cmd.text.strip() for cmd in commands if cmd.text)
return phase\_commands

# Create network topology with VLAN configuration

def create\_topology():
net = Mininet()

```
# Adding hosts with specific IP addresses
ftp = net.addHost('ftp', ip='192.168.10.2')
ssh = net.addHost('ssh', ip='192.168.20.2')
mysql = net.addHost('mysql', ip='192.168.30.2')
smtp = net.addHost('smtp', ip='192.168.40.2')

# Adding firewall as a switch with specific DPID
firewall = net.addSwitch('firewall', dpid='0000000000000001')

# Adding links between hosts and the firewall switch
net.addLink(ftp, firewall)
net.addLink(ssh, firewall)
net.addLink(mysql, firewall)
net.addLink(smtp, firewall)

# Starting the network
net.start()

# Setting up VLANs by adding virtual interfaces on firewall switch
for i in range(1, 5):
    firewall.cmd(f'vconfig add firewall-eth{i} {i * 10}')

# Bringing up VLAN interfaces on the firewall
for i in range(1, 5):
    firewall.cmd(f'ifconfig firewall-eth{i}.{i * 10} up')

# Assigning IPs to firewall VLAN interfaces
for i in range(1, 5):
    firewall.cmd(f'ifconfig firewall-eth{i}.{i * 10} 192.168.{i * 10}.1/24')

# Adding default gateway for each host to route through the firewall
for host in [ftp, ssh, mysql, smtp]:
    host.cmd(f'ip route add default via 192.168.{host.IP().split(".")[1] * 10}.1')

# Enabling IP forwarding on the firewall
firewall.cmd('sysctl -w net.ipv4.ip_forward=1')

# Adding NAT rule on firewall for external traffic (assuming eth0 is external)
firewall.cmd('iptables -t nat -A POSTROUTING -o firewall-eth0 -j MASQUERADE')

# Return the net object and hosts for further processing
return net, [ftp, ssh, mysql, smtp]
```

# Main function

if **name** == '**main**':
setLogLevel('info')
print("Creating topology and starting Q-learning process...")
phase\_commands = read\_commands\_from\_file('attack.xml')
net, hosts = create\_topology()  # Get hosts from the topology
agent\_one = AgentOne(phase\_commands)
agent\_two = AgentTwo(attack\_phases, actions, rewards, phase\_commands)   # Pass the correct parameters to AgentTwo
agent\_two.q\_learning\_process()

```
# Output learned Q-values
print("Learned Q-values:")
print(agent_two.Q)

# Stop the network after testing
net.stop()
```
