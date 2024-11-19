import random
import torch
import numpy as np

# Minesweeper 
# Create the game with game = 
# A player is allowed access to board and mines remaining variables 
# A player is allowed to use reveal_tile(x,y), flag_tile(x,y) and unflag_tile(x,y)

class Game:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.mines = mines
        self.board_size = width * height
        self.randomize_board()
        self.truncate = 0
        self.total_step = 0
        

    def board_state(self):
        if self.won:
            return "Won"
        elif self.lost:
            return "Lost"
        elif self.total_step >=  self.board_size + 10 or self.truncated > 10:
            return "Truncated"
        
    def board_to_tensor(self, board):

            board = np.array(board)
            board = torch.from_numpy(board).to(torch.float32)
            board = torch.reshape(board, (-1, ))
            #print('board')

            return board
    
    def x_y_transform(self, output_neuron_pos):
     x = output_neuron_pos % self.height
     y = output_neuron_pos % self.width

     return x, y
    
    def reset_board(self):
        self.board = self.board_copy
        board_tensor = self.board_to_tensor(self.board)
        return board_tensor
    

    
    #Resets the game
    def randomize_board(self):
        self.won = False
        self.lost = False
        self.board = [[9 for _ in range(self.width)] for _ in range(self.height)]
        
        self.hidden_board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.mines_remaining = self.mines
        self.not_mines = self.width*self.height - self.mines
        self.truncate = 0
        self.total_step = 0
        count = self.width * self.height
        i = 0
        self.mines_remaining = self.mines
        while i < self.mines:
            xpos = random.randint(0, self.width-1)
            ypos = random.randint(0, self.height-1)
            if self.hidden_board[ypos][xpos] != 'x':
                self.hidden_board[ypos][xpos] = 'x'
                i += 1
                for tile in self.get_neighbours(xpos, ypos):
                    if self.hidden_board[tile[1]][tile[0]] != 'x':
                        self.hidden_board[tile[1]][tile[0]] += 1
        
        self.board_copy = self.board
        board_tensor = self.board_to_tensor(self.board)
        return board_tensor
    
        
    #Reveals the chosen tile
    # x is distance from the left 0 to (width-1)
    # y is distance from the top
    def reveal_tile(self, x, y):
        
        if (x<self.width and y<self.height):
            if(self.board[y][x]== 9):
                hidden_value = self.hidden_board[y][x]
                self.board[y][x] = hidden_value
                if hidden_value == 'x':
                    self.lost = True
                    self.board[y][x] = 777
                    return True
                self.not_mines -= 1
                if self.not_mines == 0:
                    self.won = True
                if hidden_value == 0:
                    for tile in self.get_neighbours(x,y):
                        self.reveal_tile(tile[0], tile[1])
            self.truncate =0
            return True
        else:
            self.truncate += 1
            return False
        
    def action(self, x, y):
        self.total_step +=1
        valid_move = self.reveal_tile(x, y)
        reward = 0
        done = False
        if(self.won == True):
            reward = 1
        elif(self.lost == True or self.total_step > self.board_size + 10):
            reward = -1
        elif(valid_move == False):
            reward = - 1/ self.board_size
        else:
            reward = 1/ self.board_size
        
        if(self.won == True or self.lost == True or self.truncate >= 10 or self.total_step > self.board_size + 10):
            done = True
        else: 
            done = False

        board_tensor = self.board_to_tensor(self.board)
        return board_tensor, reward, done, 
        
    def random_action(self):
        random_x = random.randrange(0, self.width -1)
        random_y = random.randrange(0, self.height -1)
        action = random_x + random_y * self.width

        board, reward, done = self.action(random_x, random_y)
        return board, action, reward, done
    

    def flag_tile(self, x, y):
        if(self.board[y][x]==9):
            self.board[y][x] = 99
            self.mines_remaining -= 1

    def unflag_tile(self, x, y):
        if(self.board[y][x]==99):
            self.board[y][x] = 9
            self.mines_remaining += 1

    def get_neighbours(self, x,y):
        #this code sucks but it works 
        output = []
        top = y == 0
        left = x == 0
        bottom = y == (self.height-1) 
        right = x == (self.width-1)
        if not top:
            if not left:
                output.append((x-1, y-1))
            output.append((x, y-1))
            if not right:
                output.append((x+1,y-1))
        if not left:
            output.append((x-1, y))
        if not right:
            output.append((x+1, y))
        if not bottom:
            if not left:
                output.append((x-1, y+1))
            output.append((x, y+1))
            if not right:
                output.append((x+1,y+1))
        return output

    #For playing
    #Prints the board to console, currently also prints the hidden game boards
    def printGame(self):
        for row in range(self.height):
            for x in range(self.width):
                value = self.board[row][x]
                if value == 0:
                    value = '-'
                print(value, end="")
            print("    |    ", end='')
            for x in range(self.width):
                print(self.hidden_board[row][x], end="")
            print("")

        print("\n \n")
    