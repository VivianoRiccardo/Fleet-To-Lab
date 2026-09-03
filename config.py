import torch

FILES_GROUP_1 = ["./data/training_data_leo_rover_lunalab_default_friction_particle_0_1_density_0_025_size_part1.csv","./data/training_data_leo_rover_lunalab_default_friction_particle_0_1_density_0_025_size_part2.csv"]
FILES_GROUP_2 = ["./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part1.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part2.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part3.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part4.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part5.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part6.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part7.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part8.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part9.csv","./data/training_data_cadre_default_moon_default_friction_particle_0_01_density_0_0125_size_part10.csv",]
FILES_GROUP_3 = ["./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part1.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part2.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part3.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part4.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part5.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part6.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part7.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part8.csv","./data/training_data_husky_default_moon_default_friction_particle_0_01_density_0_0125_size_part9.csv"]
FILES_GROUP_4 = ["./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part1.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part2.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part3.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part4.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part5.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part6.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part7.csv","./data/training_data_leo_rover_default_moon_default_friction_particle_0_01_density_0_0125_size_part8.csv",]

BATCH_SIZE = 32
BATCH_SIZE_ADAPT = 200
LR_BASE = 0.0035
LR_ADAPT = 0.003
LR_ADAPT_ONLY = 0.003
EPOCHS_BASE = 70
EPOCHS_ADAPT = 130
ADAPT_PERCENT_FIRST_CLASS = 0.06
ADAPT_PERCENT_SECOND_CLASS = 0.06
ADAPT_PERCENT_THIRD_CLASS = 0.06


SWARM_SIZE = 1000
SWARM_ITERATIONS = 130
STEP_LAMBDA = 1.0
LAMBDA_DECAY = 0.95
INERTIA = 0.3
COGNITIVE = 0.4
SOCIAL = 0.4
REPEL = 0.1
PATIENCE = 10
RESTART_PATIENCE = 5


GA_GENERATIONS = EPOCHS_ADAPT
EXPERT_CLOUD = 15
RANDOM_CLOUD = 0
EXPERT_MULTIPLIER = EXPERT_CLOUD + RANDOM_CLOUD


NUM_CLASSES = 3
INPUT_DIM_EXPERT = 12       
INPUT_DIM_CONDITIONAL = 17 


MASTER_SEED = 12345
N_REPEATS = 30


FITNESS_CE_PENALTY = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on: {DEVICE} ---")
if torch.cuda.is_available():
    print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")
