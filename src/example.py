import soccerdata as sd
understat = sd.understat(leagues=['ENG-Premier League'], seasons=[2025])
season_stats = understat.read_player_season_stats(stat_type='shooting')