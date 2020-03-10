from ale_py import ALEInterface
import numpy as np
from random import random
from random import randint

np.set_printoptions(threshold=np.inf)
#A note on directions: Up,Right,Left,Down are 2,3,4,5 respectively, as per ALE
#Feature list and their numbering:

#0. ghosts_distance #Sum of geommetric distance from all killing ghosts
#1. ghosts_real_distance  #Sum of distance from all killing ghosts considering walls
#2. pellets_nearby  #Sum of number of pellets nearby
#3. powerup_nearby #Sum of number of remaining powerups nearby
#4. pellets_now_onbard #Remaining pellets on board
#5. dying_ghosts_distance  #Sum of geommetric distance from all dying ghosts
#6. ghosts_directions #Number of killing ghosts heading into same direction as action a
#7. current_score 
#8. current_lives 


#Start by setting parameters
feature_enabled =[0,2,3,4,5,6,7,8] #Stores which features are in use.The features' numbering as abovel whichever shows up in this array is considered enabled
seed = 123
training_episodes = 500
test_episodes = 20
frame_skip = 5
learning_rate = 0.000002
discount_factor = 0.9
epsilon = 0.2
action_selection = 1 #1 is using epsilon-greedy, 2 is using optimistic prior

def ram_to_state(ram):
    #This function takes a ram as input and extracts useful data from it. Namely positions of objects, current score, current lives and number of pellets left.
    #Structure of state: a total of 24 elements
    #Pacman X and Y,Ghosts X,Y, direction and edibility, Fruit X,Y and Existence, score, lives, #pellets left
    state = np.zeros(24,dtype = np.uint64)
    #Get position of Pacman
    state[0] = ram[10]
    state[1] = ram[16]
    for i in range(4):
        #Get position of ghosts
        state[i*4+2] = ram[i+6]
        state[i*4+3] = ram[i+12]
        #Get direction of ghosts and change them to same convention as ALE
        state[i*4+4] = ram[i+1] & 3
        if state[i*4+4] == 0: #Up in RAM
            state[i*4+4] = 2
        elif state[i*4+4] == 1: #Right in RAM
            state[i*4+4] == 3
        elif state[i*4+4] == 2: #Down in RAM
            state[i*4+4] == 5
        elif state[i*4+4] == 3: #Left in RAM
            state[i*4+4] == 4
        #Get edibility of ghosts
        if ram[i+1] >= 128:
            state[i*4+5] = 1 #1 for edible, 0 for not edible
        else:
            state[i*4+5] = 0
    #Get position of fruit
    state[18] = ram[11]
    state[19] = ram[17]
    if ram[11] >= 16: #If first digit of x coordinate of fruit is not 0, it exists
        state[20] = 1 
    else:
        state[20] = 0
    #Get current score, lives and #of pellets left on board
    state[21] = ram[122] * 10000 + ram[121] * 100 + ram[120] #Current score
    state[22] = ram[123] #Remaining lives
    state[23] = 154 - ram[119] #Pellets on board; ram[119] stores amount of pellets eaten in this board
        
    return state

def action_picking_epsilon_greedy(state,game_map,legal_actions,feature_enabled,feature_weights,epsilon):
    #This function picks an action based on current state, optimistic priors/exploration functions and whether we use lookahead search
    #This is using epsilon greedy as exploration function. 
    q_value_temp = 0
    q_value_max = 0
    i = 0
    rand = 0   
    action = 2
    for a in legal_actions:
        feature_value_temp = feature_calculation(state,a,game_map,feature_enabled)
        q_value_temp = q_calculation(feature_value_temp,feature_enabled,feature_weights)
        if q_value_temp > q_value_max:
            q_value_max = q_value_temp
            action = a
    rand = random()
    if rand < epsilon:
        action = legal_actions[randint(0,len(legal_actions)-1)]
    return action

def action_picking_optim_prior(state,game_map,legal_actions,feature_enabled,feature_weights,times_visited_table):
    #This function picks an action based on current state, optimistic priors/exploration functions and whether we use lookahead search
    #This is using ea descending function as opportunistic prior. 
    q_value_temp = 0
    q_value_max = 0
    i = 0
    rand = 0   
    action = 2
    for a in legal_actions:
        feature_value_temp = feature_calculation(state,a,game_map,feature_enabled)

        #Extract times visited
        feature_ID = str(feature_value_temp)
        if feature_ID in times_visited_table.keys():
            times_visited_table[feature_ID] += 1
        else:
            times_visited_table.update({feature_ID:1})
        q_value_temp = q_calculation(feature_value_temp,feature_enabled,feature_weights)
        q_value_temp = prior_calculation(times_visited_table[feature_ID],10,1000000,q_value_temp)
        if q_value_temp > q_value_max:
            q_value_max = q_value_temp
            action = a
    return action

def prior_calculation(times_visited,cutoff,max_prior,q_value):
    #This is the function to calculate optimistic prior as a decending function of times a state is visited
    if times_visited < cutoff:
        return max_prior
    else:
        return q_value

def update_map(state_prev,state_current,action_prev,game_map,reset):
    #This function takes previous state and current state (both as ram), then update the internal game map accordingly
    #This map will not explicitly store positions of pacman and ghosts, since we won't use this to visualize the board anyways
    #First, define the notations
    path = 0
    wall = 9
    pellet_available = 1
    pellet_eaten = 2
    powerup_available = 3
    powerup_eaten = 4
    fruit = 5
    killing_ghost = 6
    dying_ghost = 7
    pacman = 8

    #Pacman X and Y,Ghosts X,Y, direction and edibility, Fruit X,Y and Existence, score, lives, #pellets left, a total of 24
    #directions: Up,Right,Left,Down are 2,3,4,5 respectively, as per ALE

    #To change coordinates from state to our map, we do <x-17,y-1>

    #Refresh map if all pellets are eaten
    if (reset == 1) or (state_current[23] == 154): 
        game_map[game_map == 2] = 1 #Refresh pellets
        game_map[game_map == 4] = 3 #Refresh power-ups
    #Draw walls if pacman doesn't move
    if (state_prev[0] == state_current[0]) and (state_prev[1] == state_current[1]):         
        if action_prev == 2:
            game_map[state_prev[0],int(state_prev[1]-1)] = 9 #Wall on UP
        if action_prev == 3:
            game_map[int(state_prev[0]+1),state_prev[1]] = 9 #Wall on Right
        if action_prev == 4:
            game_map[int(state_prev[0]-1),state_prev[1]] = 9 #Wall on Left
        if action_prev == 5:
            game_map[state_prev[0],int(state_prev[1]+1)] = 9 #Wall on Down
    #Draw pellets if eaten
    if state_current[23] != state_prev[23]: #Number of pellets changed
        game_map[state_current[0],state_current[1]] = 2
    #Draw powerup if eaten
    if state_current[9] > state_prev[9]: #A ghost turned into edible 
        game_map[state_current[0],state_current[1]] = 4
    return game_map

def feature_calculation(state_prev,action_prev,game_map,feature_enabled):
    #This function computes the values of various features (NOT their weights!)

    #Note that here we would assume ghosts don't move; i.e. we calculate distance based on the position of pacman AFTER it takes an action
    #as well as environment BEFORE it takes that action; This computes what we "think" Q(s,a) would be, we then compare it with observed results

    feature_values = np.zeros(9,dtype = np.uint64) #A total of 9 possible features
    ghosts_distance = 0 
    ghosts_real_distance = 0 #Sum of distance from all killing ghosts considering walls
    pellets_nearby = 0 #Sum of number of pellets nearby
    powerup_nearby = 0 #Sum of number of remaining powerups nearby
    pellets_now_onbard = 154 #Remaining pellets on board
    dying_ghosts_distance = 0 #Sum of distance from all dying ghosts considering walls
    ghosts_directions = 0 #Number of killing ghosts heading into same direction as action a
    current_score = 0
    current_lives = 0

    #Update Pacman position after action
    pacman_pos = [state_prev[0],state_prev[1]]
    if (action_prev == 2) and (game_map[pacman_pos[0],int(pacman_pos[1]-1)] != 9): #If there isnt a wall in the direction of action
        pacman_pos[1] -= 1 #Move in that direction
    elif (action_prev == 3) and (game_map[int(pacman_pos[0]+1),pacman_pos[1]] != 9):
        pacman_pos[0] += 1
    elif (action_prev == 4) and (game_map[int(pacman_pos[0]-1),pacman_pos[1]] != 9):
        pacman_pos[0] -= 1
    elif (action_prev == 5) and (game_map[pacman_pos[0],int(pacman_pos[1]+1)] != 9):
        pacman_pos[1] += 1
    #Get a temporary map around pacman
    temp_map = game_map[int(pacman_pos[0]-5):int(pacman_pos[0]+6),int(pacman_pos[1]-5):int(pacman_pos[1]+6)] #Get a 6X6 submap around the pacman

    pacman_pos[0] = int(pacman_pos[0])
    pacman_pos[1] = int(pacman_pos[1])
    #0 Compute ghosts distance (not considering walls) if enabled
    if 0 in feature_enabled:    
        for i in range(4):
            if state_prev[i*4+5] == 0: #Non-edible
                ghosts_distance += abs(state_prev[i*4+2] - pacman_pos[0])
                ghosts_distance += abs(state_prev[i*4+3] - pacman_pos[1])
        feature_values[0] = ghosts_distance

    #1 Compute ghosts real distance (considering walls) if enabled
    if 1 in feature_enabled:    
        for i in range(4):
            if state_prev[i*4+5] == 0:
                #Pending
                feature_values[1] += ghosts_distance 

    #2 Compute amount of pellets nearby if enabled
    if 2 in feature_enabled:            
        pellets_nearby = np.count_nonzero(temp_map == 1)
        feature_values[2] = pellets_nearby 

    #3 Compute amount of powerup nearby if enabled
    if 3 in feature_enabled:
        powerup_nearby = np.count_nonzero(temp_map == 3)
        feature_values[3] = powerup_nearby 
    
    #4 Compute how many pellets are on board now if enabled
    if 4 in feature_enabled:
        pellets_now_onbard = state_prev[23]
        feature_values[4] = pellets_now_onbard

    #5 Compute dying ghosts distance (not considering walls) if enabled
    if 5 in feature_enabled: 
        for i in range(4):
            if state_prev[i*4+5] == 1: #edible
                dying_ghosts_distance += abs(state_prev[i*4+2] - pacman_pos[0])
                dying_ghosts_distance += abs(state_prev[i*4+3] - pacman_pos[1])
        feature_values[5] = dying_ghosts_distance 
    
    #6 Compute how many ghosts were facing same direction as action direction if enabled
    if 6 in feature_enabled: 
        for i in range(4):
            if state_prev[i*4+4] == action_prev: #Facing same direction as action
                ghosts_directions += 1
        feature_values[6] = ghosts_directions 

    #7 Compute current score if enabled
    if 7 in feature_enabled:
        current_score = state_prev[21]
        feature_values[7] = current_score 

    #8 Compute remaining lives if enabled
    if 8 in feature_enabled:
        current_lives = state_prev[22]
        feature_values[8] = current_lives 

    return feature_values

def q_calculation(feature_values,feature_enabled,feature_weights):
    q_value = 0
    #This function calculates the Q value associated with the previously executed state/action pair      
    if 0 in feature_enabled:    
        q_value += feature_values[0] * feature_weights[0]
    #1 Compute ghosts real distance (considering walls) if enabled
    if 1 in feature_enabled:    
        q_value += feature_values[1] * feature_weights[1]            
    #2 Compute amount of pellets nearby if enabled
    if 2 in feature_enabled:            
        q_value += feature_values[2] * feature_weights[2]
    #3 Compute amount of powerup nearby if enabled
    if 3 in feature_enabled:
        q_value += feature_values[3] * feature_weights[3]
    #4 Compute how many pellets are on board now if enabled
    if 4 in feature_enabled:
        q_value += feature_values[4] * feature_weights[4]
    #5 Compute dying ghosts distance (not considering walls) if enabled
    if 5 in feature_enabled: 
        q_value += feature_values[5] * feature_weights[5]         
    #6 Compute how many ghosts were facing same direction as action direction if enabled
    if 6 in feature_enabled: 
        q_value += feature_values[6] * feature_weights[6] 
    #7 Compute current score if enabled
    if 7 in feature_enabled:
        q_value += feature_values[7] * feature_weights[7] 
    #8 Compute remaining lives if enabled
    if 8 in feature_enabled:
        q_value += feature_values[8] * feature_weights[8] 
    return q_value

def update_weights(reward,feature_values_prev,state_current,feature_enabled,feature_weights,learning_rate,discount_factor,game_map):
    #Update the features weights according to estimation from previous state+action and currently observed state from action
    q_value_current = 0
    #Find Q(s,a)
    q_value_prev = q_calculation(feature_values_prev,feature_enabled,feature_weights)
    #Find max(Q(s',a'))
    for a in [2,3,4,5]:
        feature_values_current_temp = feature_calculation(state_current,a,game_map,feature_enabled)
        q_value_temp = q_calculation(feature_values_current_temp,feature_enabled,feature_weights)
        if q_value_temp > q_value_current:
            q_value_current = q_value_temp
    

    #Now, update features
    for i in range(len(feature_values_prev)):
        if i in feature_enabled:
            error = reward + discount_factor * q_value_current - q_value_prev
            #error = np.clip(error,-10000000,10000000)
            feature_weights[i] = feature_weights[i] + learning_rate*(error) * feature_values_prev[i]
            #print('Current Q is: ', q_value_current)
            #print('Error is :',error)
            #print('Previous Q is :', q_value_prev)
    return feature_weights

def draw_map(game_map):
    #This function draws out the game map, sort of...
    for i in range(160):
        for j in range(158):
            if game_map[i,j] == 9:
                print('X',end = "")
            if (game_map[i,j] == 2) or (game_map[i,j] == 1):
                print('O',end = "")
            if (game_map[i,j] == 4) or (game_map[i,j] == 3):
                print('P',end = "")
            else:
                print(" ", end = "")
        if game_map[i,158] == 9:
            print('X')
        elif (game_map[i,158] == 2) or (game_map[i,158] == 1):
            print('O')
        elif (game_map[i,158] == 4) or (game_map[i,158] == 3):
            print('P')
        else:
            print(" ")
    return 0



#Initialize ALE and relevant parameters
ale = ALEInterface()
ale.setInt("frame_skip",frame_skip)
ale.setInt("random_seed",seed)
ale.loadROM("ms_pacman.bin")
actions = ale.getMinimalActionSet()


#Here, we only use RAM as input for the agent
ram_size = ale.getRAMSize()
ram = np.zeros((ram_size),dtype=np.uint8)
state_prev = np.zeros(24,dtype=np.uint64)
state_current = np.zeros(24,dtype=np.uint64)



feature_weights = np.zeros(9) #Weight of features for function approximation currently in use
times_visited_table = dict()
#We know the game coordinates range are <x,y> = <18,2> to <???,158>; a total of ??? x values and 157 y values
#For notation of game_map, see update_map function
game_map = np.zeros((210,160),dtype = np.uint8) #We use the total amount of pixels... screw calculations
#Note that now, the topic left corner for Pacman to stay is game_map[17,1] instead of [0,0]
actions = [2,3,4,5]
for episode in range(training_episodes):
    #Update optimistic prior/exploration function
    total_reward = 0.0
    ale.getRAM(ram)
    state_prev = ram_to_state(ram)
    while not ale.game_over():       
        #pick an action and observe reward
        if action_selection == 1:
            a = action_picking_epsilon_greedy(state_prev,game_map,actions,feature_enabled,feature_weights,epsilon)     
        else:
            a = action_picking_optim_prior(state_prev,game_map,actions,feature_enabled,feature_weights,times_visited_table)
        reward = ale.act(a);

        feature_values_prev = feature_calculation(state_prev,a,game_map,feature_enabled)
        total_reward += reward

        #Get new state
        ale.getRAM(ram)
        state_current = ram_to_state(ram)
        #Update the map from observed action and reward           
        game_map = update_map(state_prev,state_current,a,game_map,0)
        #Update the feature weights
        feature_weights = update_weights(reward,feature_values_prev,state_current,feature_enabled,feature_weights,learning_rate,discount_factor,game_map)      
        state_prev = state_current
        
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    print("Episode lasted: " + str(ale.getEpisodeFrameNumber()//60) + " seconds")   
    ale.reset_game()
    game_map = update_map(state_prev,state_current,a,game_map,1)

#After training, now test with trainned model using epsilon greedy where epsilon = 0
#At this point, feature weights remain unchanged
epsilon = 0
total_score = 0
total_time = 0
scores = np.zeros(test_episodes)
times = np.zeros(test_episodes)
for episode in range(test_episodes):
    #Reset ALE with another seed
    ale = ALEInterface()
    ale.setInt("frame_skip",frame_skip)
    ale.setInt("random_seed",randint(1,200))
    ale.loadROM("ms_pacman.bin")

    total_reward = 0.0
    episode_time = 0
    ale.getRAM(ram)
    state_prev = ram_to_state(ram)
    while not ale.game_over():       
        #pick an action and observe reward
        a = action_picking_epsilon_greedy(state_prev,game_map,actions,feature_enabled,feature_weights,epsilon)  
        #a = actions[np.random.randint(4)]
        reward = ale.act(a);

        feature_values_prev = feature_calculation(state_prev,a,game_map,feature_enabled)
        total_reward += reward

        #Get new state
        ale.getRAM(ram)
        state_current = ram_to_state(ram) 
  
        state_prev = state_current
    episode_time = ale.getEpisodeFrameNumber()//60
    scores[episode] = total_reward 
    times[episode] = episode_time
    print("Test Episode " + str(episode) + " ended with score: " + str(total_reward))
    print("Test Episode lasted: " + str(episode_time) + " seconds")   
    ale.reset_game()
    game_map = update_map(state_prev,state_current,a,game_map,1)
print ("Test finished")
print ("Average score is: " + str(np.mean(scores)))
print ("Standard deviation for score is: " , np.std(scores))
print ("Average time is: " + str(np.mean(times)))
print ("Standard deviation for time is: " , np.std(times))
#report the model used and features
print(feature_weights)
#draw_map(game_map)

