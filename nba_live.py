from nba_api.live.nba.endpoints import scoreboard

# Today's Score Board
games = scoreboard.ScoreBoard()



# dictionary
lol = games.get_dict()

print(lol)

