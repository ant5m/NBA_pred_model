"use client";

import { useEffect, useState } from "react";
import { format } from "date-fns";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { TrendingUp, Award, Target, Zap, RefreshCw } from "lucide-react";
import { LoadingSpinner } from "@/components/LoadingSpinner";

interface MonthlyAccuracy {
  month: string;
  correct: number;
  total: number;
  accuracy: number;
  avg_confidence: number;
}

interface OverallAccuracy {
  total_predictions: number;
  total_correct: number;
  overall_accuracy: number;
  avg_confidence: number;
  best_month: string | null;
  best_month_accuracy: number | null;
  recent_streak: number;
}

export default function AccuracyPage() {
  const [monthlyData, setMonthlyData] = useState<MonthlyAccuracy[]>([]);
  const [overallData, setOverallData] = useState<OverallAccuracy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState(false);
  const [updateMessage, setUpdateMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchAccuracyData();
  }, []);

  const fetchAccuracyData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [monthlyRes, overallRes] = await Promise.all([
        fetch(`${process.env.NEXT_PUBLIC_API_URL}/accuracy/monthly`),
        fetch(`${process.env.NEXT_PUBLIC_API_URL}/accuracy/overall`),
      ]);

      if (!monthlyRes.ok || !overallRes.ok) {
        throw new Error("Failed to fetch accuracy data");
      }

      const monthly = await monthlyRes.json();
      const overall = await overallRes.json();

      setMonthlyData(monthly);
      setOverallData(overall);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const formatMonth = (monthStr: string) => {
    const [year, month] = monthStr.split("-");
    return format(new Date(parseInt(year), parseInt(month) - 1), "MMM yyyy");
  };

  const updateResults = async () => {
    setUpdating(true);
    setUpdateMessage(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/admin/update-results`,
        { method: "POST" }
      );

      if (!response.ok) {
        throw new Error("Failed to update results");
      }

      const data = await response.json();
      setUpdateMessage(
        `✓ ${data.message} (${data.updated} games updated)`
      );

      // Refresh accuracy data after update
      await fetchAccuracyData();
    } catch (err) {
      setUpdateMessage(
        `✗ ${err instanceof Error ? err.message : "Update failed"}`
      );
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <Award className="w-10 h-10 text-nba-blue" />
            Model Accuracy
          </h1>
        </div>
        <button
          onClick={updateResults}
          disabled={updating || loading}
          className="flex items-center gap-2 px-4 py-2 bg-nba-blue text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${updating ? 'animate-spin' : ''}`} />
          {updating ? "Updating..." : "Update Results"}
        </button>
      </div>

      {/* Update Message */}
      {updateMessage && (
        <div className={`mb-4 p-4 rounded-lg ${
          updateMessage.startsWith('✓') 
            ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200'
            : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
        }`}>
          {updateMessage}
        </div>
      )}

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

      {/* Content */}
      {!loading && !error && overallData && (
        <>
          {/* Overall Stats */}
          <div className="grid md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border-l-4 border-nba-blue">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Overall Accuracy
                </div>
                <Target className="w-5 h-5 text-nba-blue" />
              </div>
              <div className="text-3xl font-bold text-nba-blue">
                {Math.round(overallData.overall_accuracy * 100)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {overallData.total_correct} / {overallData.total_predictions}{" "}
                correct
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border-l-4 border-green-500">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Best Month
                </div>
                <Award className="w-5 h-5 text-green-500" />
              </div>
              <div className="text-3xl font-bold text-green-600">
                {overallData.best_month_accuracy
                  ? `${Math.round(overallData.best_month_accuracy * 100)}%`
                  : "N/A"}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {overallData.best_month
                  ? formatMonth(overallData.best_month)
                  : "No data"}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border-l-4 border-purple-500">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Avg Confidence
                </div>
                <TrendingUp className="w-5 h-5 text-purple-500" />
              </div>
              <div className="text-3xl font-bold text-purple-600">
                {Math.round(overallData.avg_confidence * 100)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">Model certainty</div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border-l-4 border-orange-500">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Current Streak
                </div>
                <Zap className="w-5 h-5 text-orange-500" />
              </div>
              <div className="text-3xl font-bold text-orange-600">
                {overallData.recent_streak}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Consecutive correct
              </div>
            </div>
          </div>

          {/* Monthly Accuracy Chart */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg mb-8">
            <h2 className="text-2xl font-bold mb-6">Monthly Accuracy</h2>
            {monthlyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="month"
                    tickFormatter={formatMonth}
                    stroke="#9CA3AF"
                  />
                  <YAxis
                    stroke="#9CA3AF"
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    domain={[0, 1]}
                  />
                  <Tooltip
                    formatter={(value: number) =>
                      `${(value * 100).toFixed(1)}%`
                    }
                    labelFormatter={formatMonth}
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      color: "#fff",
                    }}
                    animationDuration={200}
                  />
                  <Legend />
                  <Bar
                    dataKey="accuracy"
                    name="Accuracy"
                    radius={[8, 8, 0, 0]}
                    animationDuration={300}
                  >
                    {monthlyData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.accuracy >= 0.7
                            ? "#10B981"
                            : entry.accuracy >= 0.6
                            ? "#F59E0B"
                            : "#EF4444"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-gray-500">
                <Award className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-lg">No monthly data available yet</p>
                <p className="text-sm mt-2">
                  Make predictions and wait for games to complete
                </p>
              </div>
            )}
          </div>

          {/* Confidence Trend */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg mb-8">
            <h2 className="text-2xl font-bold mb-6">Confidence Trend</h2>
            {monthlyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="month"
                    tickFormatter={formatMonth}
                    stroke="#9CA3AF"
                  />
                  <YAxis
                    stroke="#9CA3AF"
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    domain={[0, 1]}
                  />
                  <Tooltip
                    formatter={(value: number) =>
                      `${(value * 100).toFixed(1)}%`
                    }
                    labelFormatter={formatMonth}
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      color: "#fff",
                    }}
                    animationDuration={200}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="avg_confidence"
                    stroke="#8B5CF6"
                    strokeWidth={2}
                    name="Avg Confidence"
                    dot={{ fill: "#8B5CF6", r: 4 }}
                    animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center h-[300px] text-gray-500">
                <TrendingUp className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-lg">No confidence data available yet</p>
              </div>
            )}
          </div>

          {/* Monthly Breakdown Table */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
            <h2 className="text-2xl font-bold p-6 border-b border-gray-200 dark:border-gray-700">
              Monthly Breakdown
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Month
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Correct
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Total
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Accuracy
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Avg Confidence
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {monthlyData.map((month) => (
                    <tr
                      key={month.month}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <td className="px-6 py-4 whitespace-nowrap font-medium">
                        {formatMonth(month.month)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {month.correct}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {month.total}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-semibold ${
                            month.accuracy >= 0.7
                              ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                              : month.accuracy >= 0.6
                              ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                              : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                          }`}
                        >
                          {Math.round(month.accuracy * 100)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {Math.round(month.avg_confidence * 100)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
