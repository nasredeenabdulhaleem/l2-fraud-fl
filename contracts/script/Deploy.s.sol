// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

import {Script, console2} from "forge-std/Script.sol";
import {FLAggregator} from "../src/FLAggregator.sol";

/// @notice Deploys FLAggregator and optionally pre-registers a set of client nodes.
/// @dev Usage:
///   forge script script/Deploy.s.sol:Deploy \
///     --rpc-url arbitrum_sepolia --broadcast --private-key $DEPLOYER_PRIVATE_KEY
contract Deploy is Script {
    function run() external returns (FLAggregator agg) {
        uint256 pk = vm.envUint("DEPLOYER_PRIVATE_KEY");
        vm.startBroadcast(pk);

        agg = new FLAggregator();
        console2.log("FLAggregator deployed at:", address(agg));

        // Optional: pre-register client node addresses supplied via env, comma-separated.
        // Left as a manual step here to keep the deploy deterministic and gas-transparent.

        vm.stopBroadcast();
    }
}
