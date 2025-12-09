#!/bin/bash
# Quick reference for NBA prediction model workflows

show_help() {
    echo "NBA Prediction Model - Quick Commands"
    echo "======================================"
    echo ""
    echo "DAILY WORKFLOW:"
    echo "  ./nba.sh predict        - Get today's predictions (calibrated)"
    echo "  ./nba.sh log            - Log today's predictions"
    echo "  ./nba.sh update         - Update logs with game results"
    echo "  ./nba.sh analyze        - Show performance analysis"
    echo "  ./nba.sh visualize      - Generate performance graphs"
    echo ""
    echo "MAINTENANCE:"
    echo "  ./nba.sh refresh-db     - Update database with latest games"
    echo "  ./nba.sh retrain        - Retrain model (full)"
    echo "  ./nba.sh retrain-quick  - Retrain model (fast, 50 epochs)"
    echo "  ./nba.sh calibrate      - Recalibrate from logs"
    echo ""
    echo "CLEANUP:"
    echo "  ./nba.sh cleanup        - Remove redundant files"
    echo ""
    echo "HELP:"
    echo "  ./nba.sh help           - Show this message"
    echo ""
}

case "$1" in
    predict)
        echo "🏀 Getting today's predictions (calibrated + rosters)..."
        python3.11 predict_today_calibrated.py --ensemble --rosters
        ;;
    
    log)
        echo "📝 Logging today's predictions..."
        python3.11 log_predictions.py --ensemble --calibrated
        ;;
    
    update)
        echo "🔄 Updating prediction logs with results..."
        python3.11 update_prediction_logs.py
        ;;
    
    analyze)
        echo "📊 Analyzing model performance..."
        python3.11 analyze_performance.py
        ;;
    
    visualize)
        echo "📈 Generating performance visualizations..."
        python3.11 visualize_model.py --ensemble --model ensemble_model_saved --test-size 500
        echo ""
        echo "✅ Graphs saved to visualizations/"
        ;;
    
    refresh-db)
        echo "🗄️  Updating database with latest games..."
        python3.11 build_team_stats_db.py
        ;;
    
    retrain)
        echo "🔧 Retraining model (full - this will take ~30 minutes)..."
        python3.11 retrain_and_calibrate.py
        ;;
    
    retrain-quick)
        echo "⚡ Quick retrain (50 epochs - ~10 minutes)..."
        python3.11 retrain_and_calibrate.py --quick
        ;;
    
    calibrate)
        echo "🎯 Recalibrating from prediction logs..."
        python3.11 calibrate_model.py --ensemble --from-logs
        ;;
    
    cleanup)
        echo "🧹 Cleaning up redundant files..."
        echo "This will delete:"
        echo "  - nba_live.py"
        echo "  - finetune_model.py"
        echo "  - game_rosters.py"
        echo "  - team_rosters.py"
        echo "  - build_player_stats_db.py"
        echo "  - fetch_historical_data.py"
        echo "  - run_model.sh"
        echo "  - nba_player_stats.db"
        echo ""
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f nba_live.py finetune_model.py game_rosters.py team_rosters.py
            rm -f build_player_stats_db.py fetch_historical_data.py run_model.sh
            rm -f nba_player_stats.db
            rm -f ensemble_retrain_*.log
            echo "✅ Cleanup complete!"
        else
            echo "Cancelled."
        fi
        ;;
    
    help|--help|-h|"")
        show_help
        ;;
    
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
