"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Trophy, Calendar, Award } from "lucide-react";

export default function Navigation() {
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link
            href="/"
            className="flex items-center space-x-2 hover:opacity-80 transition"
          >
            <span className="text-xl font-bold bg-gradient-to-r from-nba-blue to-nba-red bg-clip-text text-transparent">
              NBA Predictions
            </span>
          </Link>

          <div className="flex space-x-1">
            <Link
              href="/predictions"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition ${
                isActive("/predictions")
                  ? "bg-nba-blue text-white"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              }`}
            >
              <Calendar className="w-5 h-5" />
              <span className="font-medium">Predictions</span>
            </Link>

            <Link
              href="/accuracy"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition ${
                isActive("/accuracy")
                  ? "bg-nba-red text-white"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              }`}
            >
              <Award className="w-5 h-5" />
              <span className="font-medium">Accuracy</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
