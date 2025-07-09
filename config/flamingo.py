# Our custom modules.
from Kernel import Kernel
from agent.flamingo.SA_ClientAgent import SA_ClientAgent as ClientAgent
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from model.LatencyModel import LatencyModel
from util import util
from util import param

# Standard modules.
from datetime import timedelta
from math import floor
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey
import numpy as np
from os.path import exists
import pandas as pd
from sys import exit
from time import time

# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-a', '--clear_learning', action='store_true',
                    help='Learning in the clear (vs SMP protocol)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('-o', '--neighborhood_size', type=int, default=1,
                    help='Number of neighbors a client has (should only enter the multiplication factor of log(n))')
parser.add_argument('--round_time', type=int, default=10,
                    help='Fixed time the server waits for one round')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('-p', '--parallel_mode', type=bool, default=True, 
                    help='turn on parallel mode at server side')
parser.add_argument('-d', '--debug_mode', type=bool, default=False, 
                    help='print debug info')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  exit()

# Historical date to simulate.  Required even if not relevant.
historical_date = pd.to_datetime('2023-01-01')

# Requested log directory.
log_dir = args.log_dir
skip_log = args.skip_log

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
util.silent_mode = not args.verbose
num_clients = args.num_clients
neighborhood_size = args.neighborhood_size
round_time = args.round_time
num_iterations = args.num_iterations
parallel_mode = args.parallel_mode
debug_mode = args.debug_mode

if not param.assert_power_of_two(num_clients):
  raise ValueError("Number of clients must be power of 2")


print ("Silent mode: {}".format(util.silent_mode))
print ("Configuration seed: {}\n".format(seed))


# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?
kernelStopTime = midnight + pd.to_timedelta('2000:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 1000000000 * 0.1  # ns

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


### Configure the Kernel.
kernel = Kernel("Base Kernel", random_state = np.random.RandomState(seed=seed))

### Obtain random state for whatever latency model will be used.
latency_rstate = np.random.RandomState(seed=seed)

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
accy_multiplier = 100000

### What will be the scale of the shared secret?
secret_scale = 1000000


### FOR MACHINE LEARNING APPLICATIONS: LOAD DATA HERE
#
#   The data should be loaded only once (for speed).  Data should usually be
#   shuffled, split into training and test data, and passed to the client
#   parties.
#
#   X_data should be a numpy array with column-wise features and row-wise
#   examples.  y_data should contain the same number of rows (examples)
#   and a single column representing the label.
#
#   Usually this will be passed through a function to shuffle and split
#   the data into the structures expected by the PPFL clients.  For example:
#   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)
#

# For the template, we generate some random numbers that could serve as example data.
# Note that this data is unlikely to be useful for learning.  It just fills the need
# for data of a specific format.

# Randomly shuffle and split the data for training and testing.
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION



### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients

client_init_start = time()

# Iterate over all client IDs.
# Client index number starts from 1.
for i in range (a, b):
  agents.append(ClientAgent(id = i,
                name = "PPFL Client Agent {}".format(i),
                type = "ClientAgent",
                iterations = num_iterations,
                num_clients = num_clients,
                neighborhood_size = neighborhood_size,
                # multiplier = accy_multiplier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                # split_size = split_size, secret_scale = secret_scale,
                debug_mode = debug_mode,
                random_state = np.random.RandomState(seed=seed)))

agent_types.extend([ "ClientAgent" for i in range(a,b) ])


client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds = init_seconds)
print (f"Client init took {td_init}")


### Configure a service agent.

agents.extend([ ServiceAgent(
                id = b,   # set id to be a number after all clients
                name = "PPFL Service Agent",
                type = "ServiceAgent",
                random_state = np.random.RandomState(seed=seed),
                msg_fwd_delay=0,
                users = [*range(a, b)],
                iterations = num_iterations,
                round_time = pd.Timedelta(f"{round_time}s"),
                num_clients = num_clients,
                neighborhood_size = neighborhood_size,
                parallel_mode = parallel_mode,
                debug_mode = debug_mode,
                ) ])

agent_types.extend(["ServiceAgent"])


### Configure a latency model for the agents.

# Get a new-style cubic LatencyModel from the networking literature.
pairwise = (len(agent_types),len(agent_types))
model_args = { 'connected'   : True,

               # All in NYC.
               # Only matters for evaluating "real world" protocol duration,
               # not for accuracy, collusion, or reconstruction.
               'min_latency' : np.random.uniform(low = 10000000, high = 100000000, size = pairwise),
               'jitter'      : 0.3,
               'jitter_clip' : 0.05,
               'jitter_unit' : 5,
             }

model_args_de = { 'connected'   : True,

               # All in NYC.
               # Only matters for evaluating "real world" protocol duration,
               # not for accuracy, collusion, or reconstruction.
               'min_latency' : np.random.choice([3000, 12000000000], size=pairwise, p=[0.95, 0.05]),
               'jitter'      : 0.3,
               'jitter_clip' : 0.05,
               'jitter_unit' : 5,
             }

"""
- seed = 2,
- wt_flamingo_report = pd.Timedelta('10s')
- wt_flamingo_crosscheck = pd.Timedelta('5s')
- wt_flamingo_reconstruction = pd.Timedelta('5s')
- wt_flamingo_reqExtraDec = pd.Timedelta('50s')
  - clients: 256, decryptor: 30, 
    - online clients: 245, online decryptor: 28
"""

latency_model = LatencyModel ( latency_model = 'deterministic',
                              random_state = latency_rstate,
                              kwargs = model_args_de )


# Start the kernel running.
results = kernel.runner(agents = agents,
                        startTime = kernelStartTime,
                        stopTime = kernelStopTime,
                        agentLatencyModel = latency_model,
                        defaultComputationDelay = defaultComputationDelay,
                        skip_log = skip_log,
                        log_dir = log_dir)



# Print parameter summary and elapsed times by category for this experimental trial.
print()
print("########## Microbenchmarks Summary ##########")
print(f"  Protocol Iterations  : {num_iterations}")
print(f"  Number of Clients    : {num_clients}")
print(f"  Number of Decryptors : {param.committee_size}")
print("############ Experiment Setups ############")
print(f"  Vector length (Enc) : {param.vector_len:,}")
print(f"  Vector type.        : {param.vector_type}")

print()
print("----- Service Agent (Mean Time per Iteration) -----")
print(f"  - Report Step        : {results['srv_report']}")
print(f"  - Crosscheck Step    : {results['srv_crosscheck']}")
print(f"  - Reconstruction Step: {results['srv_reconstruction']}")
print(f"Total Time: {results['srv_report']+results['srv_crosscheck']+results['srv_reconstruction']}")

print()
print("----- Client Agent (Mean Time per Iteration) -----")
print(f"  - Report Step        : {results['clt_report'] / num_clients}")
print(f"  - Crosscheck Step    : {results['clt_crosscheck'] / param.committee_size}")
print(f"  - Reconstruction Step: {results['clt_reconstruction'] / param.committee_size}")
print(f"Total Time: {(results['clt_report']/num_clients) + (results['clt_crosscheck']/param.committee_size) + (results['clt_reconstruction']/param.committee_size)}")

print()
print("----- Server Agent Communication (Average per Iteration) -----")
print(f"  - Received from Clients (Report Step)       : {round(results['srv_comm_report']):,} Bytes")
print(f"  - Sent to Decryptors (Labels & Ciphertexts) : {round(results['srv_comm_label&cipher']):,} Bytes")
print(f"  - Received from Decryptors (RECON 1)        : {round(results['srv_comm_recon1']):,} Bytes")
print(f"Total Send    : {round(results['srv_comm_label&cipher']):,} Bytes.")
print(f"Total Receive : {round(results['srv_comm_report']+results['srv_comm_recon1']):,} Bytes.")

print()
print("----- Client Agent Communication (Average per Client) -----")
print(f"  - Receive from Server (Vector Download)     : {round((32/8)*param.vector_len):,} Bytes")
print(f"  - Sent to Server (Report Step)              : {round(results['clt_comm_report'] / num_clients):,} Bytes")
print("--- decryptor ---")
print(f"  - Received from Server (Labels & Cipertexts): {round(results['dec_comm_label&cipher'] / param.committee_size):,} Bytes")
print(f"  - Sent to Server (Recon1)                   : {round(results['dec_comm_recon1'] / param.committee_size):,} Bytes")
