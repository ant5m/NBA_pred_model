"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, TrendingUp, Users, Activity } from "lucide-react";
import { LoadingSpinner } from "@/components/LoadingSpinner";

interface TeamStats {
  team_id: number;
  team_name: string;
  team_abbreviation: string;
  points: number;
  field_goals_made: number;
  field_goals_attempted: number;
  field_goal_pct: number;
  three_pointers_made: number;
  three_pointers_attempted: number;
  three_point_pct: number;
  free_throws_made: number;
  free_throws_attempted: number;
  free_throw_pct: number;
  rebounds_offensive: number;
  rebounds_defensive: number;
  rebounds_total: number;
  assists: number;
  steals: number;
  blocks: number;
  turnovers: number;
  fouls_personal: number;
}

interface PlayerStats {
  player_id: number;
  player_name: string;
  team_abbreviation: string;
  position: string;
  minutes: string;
  points: number;
  field_goals_made: number;
  field_goals_attempted: number;
  field_goal_pct: number;
  three_pointers_made: number;
  three_pointers_attempted: number;
  three_point_pct: number;
  free_throws_made: number;
  free_throws_attempted: number;
  free_throw_pct: number;
  rebounds_offensive: number;
  rebounds_defensive: number;
  rebounds_total: number;
  assists: number;
  steals: number;
  blocks: number;
  turnovers: number;
  fouls_personal: number;
  plus_minus: number;
}

interface BoxScore {
  game_id: string;
  teams: TeamStats[];
  players: PlayerStats[];
}

export default function GameDetailPage() {
  const params = useParams();
  const router = useRouter();
  const gameId = params.gameId as string;

  const [boxScore, setBoxScore] = useState<BoxScore | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBoxScore();
  }, [gameId]);

  const fetchBoxScore = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/boxscore/${gameId}`
      );

      if (!response.ok) {
        throw new Error("Box score not available yet");
      }

      const data = await response.json();
      setBoxScore(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load box score");
    } finally {
      setLoading(false);
    }
  };

  const getPlayersByTeam = (teamAbbr: string) => {
    if (!boxScore) return [];
    return boxScore.players
      .filter((p) => p.team_abbreviation === teamAbbr)
      .sort((a, b) => b.points - a.points);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-nba-blue hover:underline mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Predictions
        </button>
        <h1 className="text-4xl font-bold flex items-center gap-3">
          <Activity className="w-10 h-10 text-nba-blue" />
          Game Details
        </h1>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center py-12">
          <LoadingSpinner />
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 text-center">
          <p className="text-yellow-800 dark:text-yellow-200">{error}</p>
          <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-2">
            Box score will be available once the game starts
          </p>
        </div>
      )}

      {/* Box Score */}
      {!loading && !error && boxScore && (
        <div className="space-y-8">
          {/* Team Stats Comparison */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-nba-blue to-nba-red p-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-6 h-6" />
                Team Stats
              </h2>
            </div>
            <div className="p-6">
              <div className="grid md:grid-cols-2 gap-6">
                {boxScore.teams.map((team) => (
                  <div key={team.team_id} className="space-y-3">
                    <div className="text-center mb-4">
                      <h3 className="text-2xl font-bold">{team.team_name}</h3>
                      <div className="text-4xl font-bold text-nba-blue mt-2">
                        {team.points}
                      </div>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          FG
                        </span>
                        <span className="font-semibold">
                          {team.field_goals_made}/{team.field_goals_attempted} (
                          {(team.field_goal_pct * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          3PT
                        </span>
                        <span className="font-semibold">
                          {team.three_pointers_made}/
                          {team.three_pointers_attempted} (
                          {(team.three_point_pct * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          FT
                        </span>
                        <span className="font-semibold">
                          {team.free_throws_made}/{team.free_throws_attempted} (
                          {(team.free_throw_pct * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          Rebounds
                        </span>
                        <span className="font-semibold">
                          {team.rebounds_total}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          Assists
                        </span>
                        <span className="font-semibold">{team.assists}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          Steals
                        </span>
                        <span className="font-semibold">{team.steals}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          Blocks
                        </span>
                        <span className="font-semibold">{team.blocks}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">
                          Turnovers
                        </span>
                        <span className="font-semibold">{team.turnovers}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Player Stats */}
          {boxScore.teams.map((team) => {
            const players = getPlayersByTeam(team.team_abbreviation);
            return (
              <div
                key={team.team_id}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden"
              >
                <div className="bg-gradient-to-r from-gray-700 to-gray-900 p-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                    <Users className="w-6 h-6" />
                    {team.team_name} - Player Stats
                  </h2>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-100 dark:bg-gray-700">
                      <tr className="text-xs">
                        <th className="px-4 py-3 text-left">Player</th>
                        <th className="px-2 py-3 text-center">MIN</th>
                        <th className="px-2 py-3 text-center">PTS</th>
                        <th className="px-2 py-3 text-center">REB</th>
                        <th className="px-2 py-3 text-center">AST</th>
                        <th className="px-2 py-3 text-center">STL</th>
                        <th className="px-2 py-3 text-center">BLK</th>
                        <th className="px-2 py-3 text-center">FG</th>
                        <th className="px-2 py-3 text-center">3PT</th>
                        <th className="px-2 py-3 text-center">FT</th>
                        <th className="px-2 py-3 text-center">+/-</th>
                      </tr>
                    </thead>
                    <tbody>
                      {players.map((player) => (
                        <tr
                          key={player.player_id}
                          className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <td className="px-4 py-3 font-medium">
                            {player.player_name}
                          </td>
                          <td className="px-2 py-3 text-center text-sm">
                            {player.minutes}
                          </td>
                          <td className="px-2 py-3 text-center font-semibold">
                            {player.points}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {player.rebounds_total}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {player.assists}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {player.steals}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {player.blocks}
                          </td>
                          <td className="px-2 py-3 text-center text-xs">
                            {player.field_goals_made}/
                            {player.field_goals_attempted}
                          </td>
                          <td className="px-2 py-3 text-center text-xs">
                            {player.three_pointers_made}/
                            {player.three_pointers_attempted}
                          </td>
                          <td className="px-2 py-3 text-center text-xs">
                            {player.free_throws_made}/
                            {player.free_throws_attempted}
                          </td>
                          <td
                            className={`px-2 py-3 text-center font-semibold ${
                              player.plus_minus > 0
                                ? "text-green-600"
                                : player.plus_minus < 0
                                ? "text-red-600"
                                : ""
                            }`}
                          >
                            {player.plus_minus > 0 ? "+" : ""}
                            {player.plus_minus}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
