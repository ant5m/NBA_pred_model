"use client";

import { useEffect, useState } from "react";
import { Calendar, TrendingUp, TrendingDown, ChevronRight } from "lucide-react";
import { GameCard } from "@/components/GameCard";
import { LoadingSpinner } from "@/components/LoadingSpinner";

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

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<GamePrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/predictions/today`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch predictions");
      }

      const data = await response.json();
      setPredictions(data.games || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
          <Calendar className="w-10 h-10 text-nba-blue" />
          Game Predictions
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
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 text-center">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      {/* No Games */}
      {!loading && !error && predictions.length === 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-12 text-center shadow">
          <Calendar className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <p className="text-xl text-gray-600 dark:text-gray-300">
            No games scheduled for today
          </p>
        </div>
      )}

      {/* Games Grid */}
      {!loading && !error && predictions.length > 0 && (
        <>
          <div className="mb-6 text-sm text-gray-600 dark:text-gray-400">
            Showing {predictions.length} game
            {predictions.length !== 1 ? "s" : ""} for today
          </div>

          <div className="grid gap-6">
            {predictions.map((game) => (
              <GameCard key={game.game_id} game={game} />
            ))}
          </div>
        </>
      )}

      {/* Stats Summary */}
      {!loading && predictions.length > 0 && (
        <div className="mt-8 grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Total Games
            </div>
            <div className="text-2xl font-bold">{predictions.length}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Completed
            </div>
            <div className="text-2xl font-bold">
              {predictions.filter((g) => g.actual_home_score !== null).length}
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Accuracy
            </div>
            <div className="text-2xl font-bold">
              {predictions.filter((g) => g.correct !== null).length > 0
                ? `${Math.round(
                    (predictions.filter((g) => g.correct === 1).length /
                      predictions.filter((g) => g.correct !== null).length) *
                      100
                  )}%`
                : "N/A"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
