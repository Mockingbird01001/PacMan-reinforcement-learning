# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *


import random,util,math
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf
##################################
#from keras.preprocessing.image import img_to_array
#import keras.preprocessing.image as preprocess

from PIL import Image


class QLearningAgent(ReinforcementAgent):
  
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter()
        self.position = (0,0)
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        ################ 
        ApproxState = self.transformState(state)
        
        # SARSA :
        legalActions = self.getLegalActions(state)
        max_value = None
        "*** YOUR CODE HERE ***"
        
        if util.flipCoin(self.epsilon):
              max_value = self.getQValue(ApproxState, random.choice(legalActions))
        else:
            max_value = self.getQValue(ApproxState, self.getLegalActions(state)[0])
            for action in self.getLegalActions(state)[1:] :
                  Q = self.getQValue(ApproxState, action)
                  if Q > max_value:
                        max_value = Q
        return max_value
        """         
        #Q-Learning:
        max_value = self.getQValue(ApproxState, self.getLegalActions(state)[0])
        for action in self.getLegalActions(state)[1:] :
              Q = self.getQValue(ApproxState, action)
              if Q > max_value:
                    max_value = Q
        return max_value
        """
         
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        ApproxState = self.transformState(state)
        
        if len(legalActions) == 0:
            return None
        start = random.randint(0, max(0,len(legalActions)-1))
        max_action = legalActions[start]
        max_action_value = self.getQValue(ApproxState, legalActions[start])
        for action in legalActions[0:]:
              Q = self.getQValue(ApproxState, action)
              if Q > max_action_value:
                    max_action_value = Q
                    max_action = action
        return max_action 

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if util.flipCoin(self.epsilon):
              action = random.choice(legalActions)
              #self.epsilon + self.epsilon *0.95
        else:
              action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ApproxState = self.transformState(state)
        self.Qvalues[(ApproxState, action)] = self.getQValue(ApproxState, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(ApproxState,action) )

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def transformState(self, state):
      #print(state.data.__dict__)
      # Note: nous utilisons l'algorithme de pathfinding A* de l'article
      """
      state est actuellement un truc comme ca:
      %%%%%%%%
      %P     %
      % .% . %
      %  %   %
      % .% . %
      %     G%
      %%%%%%%%
      % est le mur
      G les fantomes
      . la nourriture
      Pac-man est ^, <, > ou v
      """

      

      # print(self.Qvalues)
      # transformer le state qui est en format bizarre en un tableau a deux dimensions
      statetab = [list(s) for s in state.__str__().split('\n')[:-2]]

      self.print_state(statetab)

      # recuperer les coordonnees de pac-man, des fantomes et des capsules qu'il reste sur la map
      PacManX, PacManY, ghosts, capsules = self.getPosGame(statetab)
      self.position = (PacManX, PacManY)

      NewState = []

      # liste des cases adjacentes a pac-man
      lvoisins = [(PacManX-1,PacManY), (PacManX,PacManY-1), (PacManX+1,PacManY), (PacManX,PacManY+1)]

      # s1 a s4, check si mur
      for i in lvoisins:
        NewState.append(1 if statetab[i[0]][i[1]] == "%" else 0)
      
      # check les cases adjacentes et remplace les coordonnes des cases innacessibles par 0
      check_enemy = [x if x[0] >= 0 and x[0] < len(statetab) and x[1] >= 0 and x[1] < len(statetab[0]) and statetab[x[0]][x[1]] != '%' else 0 for x in lvoisins]

      # s5
      NewState.append(self.get5(statetab, check_enemy[:], ghosts, capsules, state)) # [:] pour en obtenir une copie

      # s6 a s9
      for direction in check_enemy:
        if direction != 0:
          for pos_ghost in ghosts:
            NewState.append(self.enemy_near(statetab, direction, pos_ghost, 8))
            break
        else:
          NewState.append(0)

      # s10
      NewState.append(self.get10(statetab, (PacManX, PacManY)))
      
      # on retourne un tuple pour notre etat approximatif
      return tuple(NewState)
      
    def getPosGame(self, state):
      # Parcourt la liste et renvoie les coordonnees interressantes
      PacX, PacY = 0, 0
      PacMan = ['^', '<', '>', 'v']
      g = []
      cap = []
      for x in range(len(state)):
        for y in range(len(state[0])):
          if state[x][y] in PacMan:
            PacX = x
            PacY = y
          elif state[x][y] == "G":
            g.append((x,y))
          elif state[x][y] == "." or state[x][y] == "o":
            cap.append((x,y))
      return (PacX, PacY, g, cap)

    def get5(self, state, list_check, ghosts, capsules, object_state):

      
      near_tab = []
      # donne une liste de la distance d'un fantome pour chaque action
      for new_pos in list_check:
        local = 8
        if new_pos != 0:
          for p_ghost in ghosts:
            path = self.a_star(state, new_pos, p_ghost)
            if path is not None and len(path) < local:
              local = len(path)
        near_tab.append(local)
        
      # si fantomes mangeables, on va vers lui si il est < 5 cases
      scaredTimer = object_state.data.agentStates[1].scaredTimer
      if scaredTimer > 0:
        if min(near_tab) < 6:
          return near_tab.index(min(near_tab))
        
      #print(list_check)
      #print(list_check.count(0))
      if list_check.count(0) > 3:
        return [x for x in range(4) if list_check[x] != 0][0]

      # supprime l'action qui mene au fantome le plus proche
      if min(near_tab) != 8:
        for i in range(4 - list_check.count(0)):
          for i in range(4):
            if near_tab[i] == min(near_tab):
              
              if list_check.count(0) == 3:
                #self.print_state(state)
                return [x for x in range(4) if list_check[x] != 0][0]
              list_check[i] = 0
              near_tab[i] = 8


      near_tab = []
      for new_pos in list_check:
        local = 9999
        if new_pos != 0:
          for pos_capsule in capsules:
            path = self.a_star(state, new_pos, pos_capsule, G = True)
            if path is not None and len(path) < local:
              local = len(path)
        near_tab.append(local)

      # si une action est + proche que les autres de nourriture, la prend
      # si plusieurs actions sont a egalite, prend au hasard
      if near_tab.count(min(near_tab)) > 1:
        return random.choice([i for i,d in enumerate(near_tab) if d== min(near_tab)])
      else:
        return near_tab.index(min(near_tab))
      
    def print_state(self, state):
      for i in state:
        print(i)

    def get10(self, state, pos):
      pos_tab = [(1,1),(len(state)-2,1),(1,len(state[0])-2),(len(state)-2,len(state[0])-2)]
      compteur = 0
      for check in pos_tab:
        if self.a_star(state, pos, check, G = True) != None:
          compteur += 1
          if compteur > 1:
            return 0
      return 1

    def enemy_near(self, state, pos, enemy_pos, max_range):
      if enemy_pos == pos:
        return 1
      path = self.a_star(state, pos, enemy_pos)
      if path is not None and len(path) < max_range:
          return 1
      return 0

    def a_star(self, grid, start, goal, G = False):
      # Initialise une liste fermee vide et une liste ouverte avec le noeud de depart
      closed_set = set()
      open_set = {start}
      
      # Initialise une liste de predecesseurs pour construire le chemin final
      came_from = {}
      
      # Initialise les valeurs g et f pour le noeud de depart
      g_score = {start: 0}
      f_score = {start: self.dist_between(start, goal)}
      
      # Boucle tant que la liste ouverte n'est pas vide
      while open_set:
          # Trouve le noeud avec la valeur f la plus faible dans la liste ouverte
          current = min(open_set, key=f_score.get)
          
          # Si le noeud actuel est le noeud final, reconstruit et renvoie le chemin
          if current == goal:
              return self.reconstruct_path(came_from, current)
          
          # Retire le noeud actuel de la liste ouverte et l'ajoute a la liste fermee
          open_set.remove(current)
          closed_set.add(current)
          
          # Pour chaque voisin du noeud actuel...
          for neighbor in self.getVoisinsValides(grid, current, G):
              # Si le voisin est deja dans la liste fermee, ignore-le
              if neighbor in closed_set:
                  continue
                  
              # Calcul la distance tentee pour atteindre ce voisin
              tentative_g_score = g_score[current] + self.dist_between(current, neighbor)
              
              # Si le voisin n'est pas dans la liste ouverte ou si la distance tentee est meilleure que la distance enregistree pour ce voisin...
              if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                  # Met a jour le predecesseur du voisin et les valeurs g et f
                  came_from[neighbor] = current
                  g_score[neighbor] = tentative_g_score
                  f_score[neighbor] = g_score[neighbor] + self.dist_between(neighbor, goal)
                  
                  # Ajoute le voisin a la liste ouverte s'il n'y est pas deja
                  if neighbor not in open_set:
                      open_set.add(neighbor)
                      
      # Si aucun chemin n'a ete trouve, renvoie None
      return None

    def getVoisinsValides(self, state, pos, G = False):
        x,y = pos
        results = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        if G == True:
          results = [x for x in results if x[0] >= 0 and x[0] < len(state) and x[1] >= 0 and x[1] < len(state[0]) and state[x[0]][x[1]] != '%' and state[x[0]][x[1]] != 'G' and (x[0],x[1]) != self.position]
          return results
        results = [x for x in results if x[0] >= 0 and x[0] < len(state) and x[1] >= 0 and x[1] < len(state[0]) and state[x[0]][x[1]] != '%' and (x[0],x[1]) != self.position]
        return results

    def dist_between(self, a, b):
      x1, y1 = a
      x2, y2 = b
      return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, came_from, current):
      path = [current]
      while current in came_from:
          current = came_from[current]
          path.append(current)
      path.reverse()
      return path

class PacmanQAgent(QLearningAgent):
  

    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        print("approxultime")
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        featureVector = self.featExtractor.getFeatures(state, action)
        Q = self.weights * featureVector
        return Q
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    
    """self.Qvalues[(ApproxState, action)] = 
    self.getQValue(ApproxState, action) 
    + self.alpha 
    * (reward 
    + self.discount * self.computeValueFromQValues(nextState) 
    - self.getQValue(ApproxState,action) )"""

    def update(self, state, action, nextState, reward):
        featureVector = self.featExtractor.getFeatures(state, action)
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        for feature in featureVector:
            self.weights[feature] += self.alpha * diff * featureVector[feature]
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        ################
        """
        # SARSA :
        legalActions = self.getLegalActions(state)
        max_value = None
        "*** YOUR CODE HERE ***"
        
        if util.flipCoin(self.epsilon):
              max_value = self.getQValue(state, random.choice(legalActions))
        else:
            max_value = self.getQValue(state, self.getLegalActions(state)[0])
            for action in self.getLegalActions(state)[1:] :
                  Q = self.getQValue(state, action)
                  if Q > max_value:
                        max_value = Q
        return max_value
        """         
        #Q-Learning:
        max_value = self.getQValue(state, self.getLegalActions(state)[0])
        for action in self.getLegalActions(state)[1:] :
              Q = self.getQValue(state, action)
              if Q > max_value:
                    max_value = Q
        return max_value
        
        
        
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        
        if len(legalActions) == 0:
            return None
        start = random.randint(0, max(0,len(legalActions)-1))
        max_action = legalActions[start]
        max_action_value = self.getQValue(state, legalActions[start])
        for action in legalActions[0:]:
              Q = self.getQValue(state, action)
              if Q > max_action_value:
                    max_action_value = Q
                    max_action = action
        return max_action 

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class PacmanDEEPQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.nb_action = 4
        self.state_size = 25
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)
        self.newstate = np.array([])

    def build_model(self):
      model = Sequential()
      model.add(Conv2D(8, (3, 3), input_shape=(7, 7, 3), activation='relu'))
      model.add(Conv2D(16, (3, 3), activation='relu'))
      model.add(Conv2D(32, (3, 3), activation='relu'))
      model.add(Flatten())
      model.add(Dense(256, activation='relu'))
      model.add(Dense(4, activation='softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      return model

    def remember(self, state, action, nextState, reward):
      """#Q-Learning:
        max_value = self.getQValue(state, self.getLegalActions(state)[0])
        for action in self.getLegalActions(state)[1:] :
              Q = self.getQValue(state, action)
              if Q > max_value:
                    max_value = Q
        return max_value"""
      
      #self.statetuple = tuple(chain.from_iterable(zip(*[list(s) for s in state.__str__().split('\n')[:-2]])))
      print('remember')
      if len(self.getLegalActions(state)) == 0:
        self.memory.append((self.newstate, action, nextState, reward, True))
      else:
        self.memory.append((self.newstate, action, nextState, reward, False))

    def replay(self, batch_size):
      print('replay')
      minibatch = random.sample(self.memory, batch_size)
      for state, action, nextState, reward, done in minibatch:
        target = reward
        if not done:
          newNextState = self.getNewState(nextState)
          target = {reward + self.gamma * np.amax(self.model.predict(newNextState)[0])}
        
        target_f = self.model.predict(self.newState)
        target_f[0][action] = target

        self.model.fit(self.newState, target_f, epochs = 1, verbose = 0)

    def load(self, name):
      self.model.load_weights(name)

    def save(self, name):
      self.model.save_weights(name)

    def update(self, state, action, nextState, reward):
        print('update')
        self.remember(state, action, nextState, reward)

        """self.Qvalues[(state, action)] = self.getQValue(state, action) 
        + self.alpha * (reward 
        + self.discount * self.computeValueFromQValues(nextState) 
        - self.getQValue(state,action) )"""
    
    def getAction(self, state):
        print('action')
        self.newstate = self.getNewState(state)
        print(self.newstate)
        if util.flipCoin(self.epsilon):
              action = random.choice(self.getLegalActions(state))
        else:
              action = self.model.predict(self.newstate) #self.getPolicy(state)

        print(action)
        self.doAction(state,action)
        return action
        
    def getNewState(self, state):
      statetab = [list(s) for s in state.__str__().split('\n')[:-2]]
      image = Image.new("RGB", (len(statetab[0]), len(statetab)),(255,255,255))
      for i in range(len(statetab)):
        for j in range(len(statetab[0])):
            if statetab[i][j] == '%':
                image.putpixel((j,i), (0,0,255))
            elif statetab[i][j] == 'G':
                image.putpixel((j,i), (0,255,255))
            elif statetab[i][j] == '>' or statetab[i][j] == '>' or statetab[i][j] == '^' or statetab[i][j] == 'v':
                image.putpixel((j,i), (255,255,0))
            elif statetab[i][j] == '.':
                image.putpixel((j,i), (255,255,255))
            else:
                image.putpixel((j,i), (0,0,0))
      img_array = tf.keras.utils.img_to_array(image)
      return img_array

    """def computeValueFromQValues(self, state):
        if len(self.getLegalActions(state)) == 0:
            return 0.0

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        
        if len(legalActions) == 0:
            return None
        start = random.randint(0, max(0,len(legalActions)-1))
        max_action = legalActions[start]
        max_action_value = self.getQValue(state, legalActions[start])
        for action in legalActions[0:]:
              Q = self.getQValue(state, action)
              if Q > max_action_value:
                    max_action_value = Q
                    max_action = action
        return max_action 

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
"""