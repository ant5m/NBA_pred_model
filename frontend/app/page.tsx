"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Trophy } from "lucide-react";

export default function Home() {
  const [stats, setStats] = useState({
    accuracy: 0,
    totalGames: 0,
    modelCount: 5,
  });

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/accuracy/overall`
      );

      if (response.ok) {
        const data = await response.json();
        setStats({
          accuracy: Math.round(data.accuracy * 100),
          totalGames: data.total_predictions,
          modelCount: 5,
        });
      }
    } catch (err) {
      console.error("Failed to fetch stats:", err);
      // Keep default values
    }
  };
  return (
    <div className="flex items-center justify-center px-4 py-16">
      <div className="max-w-4xl w-full text-center">
        <div className="flex justify-center mb-8"></div>

        <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-nba-blue to-nba-red bg-clip-text text-transparent">
          NBA Predictions
        </h1>

        <p className="text-xl text-gray-600 dark:text-gray-300 mb-12 max-w-2xl mx-auto">
          Machine learning powered predictions for NBA games. Our ensemble model
          analyzes team stats, recent performance, and historical data to
          forecast game outcomes with high accuracy.
        </p>

        <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          <Link
            href="/predictions"
            className="group bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all border-2 border-transparent hover:border-nba-blue"
          >
            <h2 className="text-2xl font-bold mb-3 group-hover:text-nba-blue transition-colors">
              Today&apos;s Predictions
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              View predictions for upcoming games with win probabilities and
              projected scores
            </p>
          </Link>

          <Link
            href="/accuracy"
            className="group bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all border-2 border-transparent hover:border-nba-red"
          >
            <h2 className="text-2xl font-bold mb-3 group-hover:text-nba-red transition-colors">
              Model Accuracy
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              Track model performance over time with monthly accuracy metrics
              and trends
            </p>
          </Link>
        </div>

        <div className="mt-16 grid grid-cols-3 gap-8 max-w-3xl mx-auto">
          <div>
            <div className="text-4xl font-bold text-nba-blue mb-2">
              {stats.accuracy}%
            </div>
            <div className="text-gray-600 dark:text-gray-300">
              Overall Accuracy
            </div>
          </div>
          <div>
            <div className="text-4xl font-bold text-nba-blue mb-2">
              {stats.totalGames}+
            </div>
            <div className="text-gray-600 dark:text-gray-300">
              Games Analyzed
            </div>
          </div>
          <div>
            <div className="text-4xl font-bold text-nba-blue mb-2">
              {stats.modelCount}
            </div>
            <div className="text-gray-600 dark:text-gray-300">
              Model Ensemble
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
