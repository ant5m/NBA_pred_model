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
  const [cachedDate, setCachedDate] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = () => {
    // Try to load from cache first
    const cached = getCachedPredictions();
    if (cached) {
      setPredictions(cached.games);
      setCachedDate(cached.date);
      setLoading(false);

      // Check if cache is stale (different date or old)
      const today = new Date().toISOString().split("T")[0];
      const cacheAge = Date.now() - cached.timestamp;
      const isStale = cached.date !== today || cacheAge > 3600000; // 1 hour

      if (isStale) {
        // Fetch in background to update cache
        fetchPredictions(true);
      }
    } else {
      // No cache, fetch immediately
      fetchPredictions();
    }
  };

  const getCachedPredictions = () => {
    try {
      const cached = localStorage.getItem("nba_predictions");
      if (cached) {
        return JSON.parse(cached);
      }
    } catch (e) {
      console.error("Failed to load cached predictions:", e);
    }
    return null;
  };

  const cachePredictions = (games: GamePrediction[], date: string) => {
    try {
      const cache = {
        games,
        date,
        timestamp: Date.now(),
      };
      localStorage.setItem("nba_predictions", JSON.stringify(cache));
    } catch (e) {
      console.error("Failed to cache predictions:", e);
    }
  };

  const fetchPredictions = async (background = false) => {
    if (!background) {
      setLoading(true);
    } else {
      setIsRefreshing(true);
    }
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
      setCachedDate(data.date);

      // Cache the predictions
      cachePredictions(data.games || [], data.date);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <Calendar className="w-10 h-10 text-nba-blue" />
            Game Predictions
          </h1>
          <button
            onClick={() => fetchPredictions()}
            disabled={isRefreshing}
            className="px-4 py-2 bg-nba-blue text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isRefreshing ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Updating...
              </>
            ) : (
              <>Refresh</>
            )}
          </button>
        </div>
        {cachedDate && (
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            Predictions for {cachedDate}
          </p>
        )}
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
