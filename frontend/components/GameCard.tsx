import { CheckCircle, XCircle, Clock, ExternalLink } from "lucide-react";
import Link from "next/link";

interface GamePrediction {
  date: string;
  game_id: string;
  home_team: string;
  away_team: string;
  predicted_home_prob: number;
  predicted_away_prob: number;
  predicted_home_score: number;
  predicted_away_score: number;
  actual_home_score: number | null;
  actual_away_score: number | null;
  actual_home_win: number | null;
  correct: number | null;
}

export function GameCard({ game }: { game: GamePrediction }) {
  const predictedWinner =
    game.predicted_home_prob >= 0.5 ? game.home_team : game.away_team;
  const predictedLoser =
    game.predicted_home_prob >= 0.5 ? game.away_team : game.home_team;
  const confidence = Math.max(
    game.predicted_home_prob,
    game.predicted_away_prob
  );

  const hasResult = game.actual_home_score !== null;
  const isCorrect = game.correct === 1;

  return (
    <Link href={`/game/${game.game_id}`}>
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow cursor-pointer">
        {/* Header with status */}
        <div className="bg-gradient-to-r from-nba-blue to-nba-red p-3 flex justify-between items-center">
          <div className="text-white text-sm font-semibold">{game.game_id}</div>
          {hasResult && (
            <div className="flex items-center space-x-1">
              {isCorrect ? (
                <>
                  <CheckCircle className="w-4 h-4 text-green-300" />
                  <span className="text-green-300 text-sm font-semibold">
                    Correct
                  </span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-red-300" />
                  <span className="text-red-300 text-sm font-semibold">
                    Incorrect
                  </span>
                </>
              )}
            </div>
          )}
          {!hasResult && (
            <div className="flex items-center space-x-1">
              <Clock className="w-4 h-4 text-yellow-300" />
              <span className="text-yellow-300 text-sm font-semibold">
                Pending
              </span>
            </div>
          )}
        </div>

        <div className="p-6">
          {/* Teams */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            {/* Away Team */}
            <div className="text-center">
              <div
                className={`text-2xl font-bold mb-2 ${
                  predictedWinner === game.away_team
                    ? "text-nba-blue"
                    : "text-gray-400"
                }`}
              >
                {game.away_team}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                Away
              </div>
              {hasResult && (
                <div className="text-3xl font-bold">
                  {game.actual_away_score}
                </div>
              )}
            </div>

            {/* VS */}
            <div className="flex flex-col items-center justify-center">
              <div className="text-gray-400 text-sm mb-2">@</div>
              <div className="text-gray-500 dark:text-gray-400 text-xs">
                Confidence: {Math.round(confidence * 100)}%
              </div>
            </div>

            {/* Home Team */}
            <div className="text-center">
              <div
                className={`text-2xl font-bold mb-2 ${
                  predictedWinner === game.home_team
                    ? "text-nba-blue"
                    : "text-gray-400"
                }`}
              >
                {game.home_team}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                Home
              </div>
              {hasResult && (
                <div className="text-3xl font-bold">
                  {game.actual_home_score}
                </div>
              )}
            </div>
          </div>

          {/* Prediction */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Predicted Outcome
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  Win Probability
                </div>
                <div className="flex justify-between text-sm">
                  <span
                    className={
                      game.predicted_away_prob > 0.5 ? "font-bold" : ""
                    }
                  >
                    {Math.round(game.predicted_away_prob * 100)}%
                  </span>
                  <span className="text-gray-400">|</span>
                  <span
                    className={
                      game.predicted_home_prob > 0.5 ? "font-bold" : ""
                    }
                  >
                    {Math.round(game.predicted_home_prob * 100)}%
                  </span>
                </div>
              </div>

              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  Projected Score
                </div>
                <div className="text-sm font-semibold">
                  {Math.round(game.predicted_away_score)} -{" "}
                  {Math.round(game.predicted_home_score)}
                </div>
              </div>
            </div>

            <div className="text-center">
              <span className="inline-block px-4 py-2 bg-nba-blue text-white rounded-full text-sm font-semibold">
                Predicted Winner: {predictedWinner}
              </span>
            </div>
          </div>
        </div>

        {/* View Details Indicator */}
        <div className="bg-gray-50 dark:bg-gray-700 px-6 py-3 flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-nba-blue dark:hover:text-nba-blue transition-colors">
          <ExternalLink className="w-4 h-4" />
          View Box Score
        </div>
      </div>
    </Link>
  );
}
