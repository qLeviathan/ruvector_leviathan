#!/bin/bash
#
# memory_coordinator.sh - Memory coordination for cross-agent FPGA testing
#
# This script manages shared memory for Claude-Flow swarm coordination,
# enabling agents to share test results, configurations, and findings.
#
# Usage:
#   ./memory_coordinator.sh store <key> <value>
#   ./memory_coordinator.sh retrieve <key>
#   ./memory_coordinator.sh list [namespace]
#   ./memory_coordinator.sh clear <key>
#

set -e

# Configuration
NAMESPACE="${FPGA_TEST_NAMESPACE:-coordination}"
SESSION_ID="${FPGA_TEST_SESSION_ID:-default}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[Memory]${NC} $*"
}

log_info() {
    echo -e "${BLUE}[Info]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[Warning]${NC} $*"
}

# ============================================================================
# Memory Operations
# ============================================================================

memory_store() {
    local key="$1"
    local value="$2"

    log "Storing: $key"

    npx claude-flow@alpha memory store \
        --key "$key" \
        --namespace "$NAMESPACE" \
        --value "$value"

    log_info "Stored successfully: $key"
}

memory_retrieve() {
    local key="$1"

    log "Retrieving: $key"

    npx claude-flow@alpha memory retrieve \
        --key "$key" \
        --namespace "$NAMESPACE"
}

memory_list() {
    local ns="${1:-$NAMESPACE}"

    log "Listing keys in namespace: $ns"

    npx claude-flow@alpha memory list \
        --namespace "$ns"
}

memory_clear() {
    local key="$1"

    log "Clearing: $key"

    npx claude-flow@alpha memory delete \
        --key "$key" \
        --namespace "$NAMESPACE"

    log_info "Cleared: $key"
}

# ============================================================================
# High-Level Operations
# ============================================================================

store_test_config() {
    local model="$1"
    local quantization="$2"
    local test_count="$3"

    log "Storing test configuration..."

    local config=$(cat <<EOF
{
    "session_id": "$SESSION_ID",
    "model": "$model",
    "quantization": "${quantization}bit",
    "test_count": $test_count,
    "timestamp": $(date +%s),
    "stored_by": "memory_coordinator"
}
EOF
)

    memory_store "swarm/fpga/test-config" "$config"
}

store_synthesis_results() {
    local netlist_path="$1"
    local luts_used="$2"
    local max_freq="$3"

    log "Storing synthesis results..."

    local results=$(cat <<EOF
{
    "session_id": "$SESSION_ID",
    "agent": "synthesis-agent",
    "netlist_path": "$netlist_path",
    "luts_used": $luts_used,
    "max_freq_mhz": $max_freq,
    "status": "success",
    "timestamp": $(date +%s)
}
EOF
)

    memory_store "swarm/synthesis/results" "$results"
}

store_test_results() {
    local results_file="$1"

    if [ ! -f "$results_file" ]; then
        log_warning "Results file not found: $results_file"
        return 1
    fi

    log "Storing test results from: $results_file"

    local results=$(cat "$results_file")

    memory_store "swarm/fpga/test-results" "$results"

    # Extract key metrics for quick access
    local accuracy=$(echo "$results" | jq -r '.accuracy // 0')
    local latency=$(echo "$results" | jq -r '.mean_latency_ms // 0')
    local throughput=$(echo "$results" | jq -r '.throughput_fps // 0')

    local summary=$(cat <<EOF
{
    "session_id": "$SESSION_ID",
    "accuracy": $accuracy,
    "mean_latency_ms": $latency,
    "throughput_fps": $throughput,
    "timestamp": $(date +%s),
    "full_results_key": "swarm/fpga/test-results"
}
EOF
)

    memory_store "swarm/fpga/test-summary" "$summary"
}

store_performance_analysis() {
    local analysis_file="$1"

    if [ ! -f "$analysis_file" ]; then
        log_warning "Analysis file not found: $analysis_file"
        return 1
    fi

    log "Storing performance analysis from: $analysis_file"

    local analysis=$(cat "$analysis_file")

    memory_store "swarm/fpga/performance-analysis" "$analysis"
}

retrieve_all_results() {
    log "Retrieving all test results..."

    echo ""
    echo "=== Test Configuration ==="
    memory_retrieve "swarm/fpga/test-config" | jq '.'

    echo ""
    echo "=== Synthesis Results ==="
    memory_retrieve "swarm/synthesis/results" | jq '.'

    echo ""
    echo "=== Test Summary ==="
    memory_retrieve "swarm/fpga/test-summary" | jq '.'

    echo ""
    echo "=== Full Test Results ==="
    memory_retrieve "swarm/fpga/test-results" | jq '.total_tests, .passed, .failed, .accuracy'
}

export_session_data() {
    local output_file="${1:-session_export.json}"

    log "Exporting session data to: $output_file"

    local export_data=$(cat <<EOF
{
    "session_id": "$SESSION_ID",
    "export_timestamp": $(date +%s),
    "test_config": $(memory_retrieve "swarm/fpga/test-config" 2>/dev/null || echo '{}'),
    "synthesis_results": $(memory_retrieve "swarm/synthesis/results" 2>/dev/null || echo '{}'),
    "test_summary": $(memory_retrieve "swarm/fpga/test-summary" 2>/dev/null || echo '{}'),
    "test_results": $(memory_retrieve "swarm/fpga/test-results" 2>/dev/null || echo '{}'),
    "performance_analysis": $(memory_retrieve "swarm/fpga/performance-analysis" 2>/dev/null || echo '{}')
}
EOF
)

    echo "$export_data" | jq '.' > "$output_file"

    log_info "Session data exported to: $output_file"
}

# ============================================================================
# Agent Coordination
# ============================================================================

notify_agent() {
    local agent_id="$1"
    local message="$2"

    log "Notifying agent: $agent_id"

    npx claude-flow@alpha hooks notify \
        --agent "$agent_id" \
        --message "$message"
}

broadcast_test_completion() {
    local accuracy="$1"
    local latency="$2"

    log "Broadcasting test completion to all agents..."

    local message="Test completed: Accuracy=$accuracy%, Latency=${latency}ms"

    npx claude-flow@alpha hooks notify \
        --broadcast \
        --message "$message"
}

# ============================================================================
# Main Command Handler
# ============================================================================

show_usage() {
    cat << USAGE
Memory Coordinator for FPGA Testing

Usage:
    $0 <command> [arguments]

Basic Commands:
    store <key> <value>                 Store value in memory
    retrieve <key>                      Retrieve value from memory
    list [namespace]                    List all keys in namespace
    clear <key>                         Clear value from memory

High-Level Commands:
    store-config <model> <quant> <count>    Store test configuration
    store-synthesis <path> <luts> <freq>    Store synthesis results
    store-test-results <file>               Store test results from JSON file
    store-performance <file>                Store performance analysis
    retrieve-all                            Retrieve all test results
    export-session [file]                   Export session data to JSON

Agent Coordination:
    notify <agent> <message>                Notify specific agent
    broadcast-completion <acc> <latency>    Broadcast test completion

Environment Variables:
    FPGA_TEST_NAMESPACE    Memory namespace (default: coordination)
    FPGA_TEST_SESSION_ID   Session ID (default: default)

Examples:
    # Store test configuration
    $0 store-config mnist_cnn 8 1000

    # Store test results
    $0 store-test-results test_results.json

    # Retrieve all results
    $0 retrieve-all

    # Export session data
    $0 export-session fpga_session_data.json

    # Broadcast completion
    $0 broadcast-completion 91.5 45.2
USAGE
}

main() {
    if [ $# -lt 1 ]; then
        show_usage
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        store)
            if [ $# -lt 2 ]; then
                echo "Error: store requires <key> <value>"
                exit 1
            fi
            memory_store "$1" "$2"
            ;;

        retrieve)
            if [ $# -lt 1 ]; then
                echo "Error: retrieve requires <key>"
                exit 1
            fi
            memory_retrieve "$1"
            ;;

        list)
            memory_list "$@"
            ;;

        clear)
            if [ $# -lt 1 ]; then
                echo "Error: clear requires <key>"
                exit 1
            fi
            memory_clear "$1"
            ;;

        store-config)
            if [ $# -lt 3 ]; then
                echo "Error: store-config requires <model> <quantization> <test_count>"
                exit 1
            fi
            store_test_config "$1" "$2" "$3"
            ;;

        store-synthesis)
            if [ $# -lt 3 ]; then
                echo "Error: store-synthesis requires <netlist_path> <luts> <freq>"
                exit 1
            fi
            store_synthesis_results "$1" "$2" "$3"
            ;;

        store-test-results)
            if [ $# -lt 1 ]; then
                echo "Error: store-test-results requires <file>"
                exit 1
            fi
            store_test_results "$1"
            ;;

        store-performance)
            if [ $# -lt 1 ]; then
                echo "Error: store-performance requires <file>"
                exit 1
            fi
            store_performance_analysis "$1"
            ;;

        retrieve-all)
            retrieve_all_results
            ;;

        export-session)
            export_session_data "$@"
            ;;

        notify)
            if [ $# -lt 2 ]; then
                echo "Error: notify requires <agent> <message>"
                exit 1
            fi
            notify_agent "$1" "$2"
            ;;

        broadcast-completion)
            if [ $# -lt 2 ]; then
                echo "Error: broadcast-completion requires <accuracy> <latency>"
                exit 1
            fi
            broadcast_test_completion "$1" "$2"
            ;;

        help|--help|-h)
            show_usage
            ;;

        *)
            echo "Error: Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
