"""Fetch AWS EC2 on-demand pricing from the public AWS offer files."""

from __future__ import annotations

import json
import math
import urllib.request
from functools import lru_cache

from ner_study.config import CostConfig
from ner_study.schemas import AwsInstanceEstimate


PUBLIC_AWS_INDEX_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json"
PUBLIC_AWS_BASE_URL = "https://pricing.us-east-1.amazonaws.com"
FALLBACK_US_EAST_1_LINUX_ON_DEMAND = {
    "c7i.large": 0.08925,
    "c7i.xlarge": 0.1785,
    "g5.xlarge": 1.006,
    "g6e.12xlarge": 10.49264,
}


@lru_cache(maxsize=8)
def fetch_ec2_region_offer_url(region: str) -> str:
    with urllib.request.urlopen(PUBLIC_AWS_INDEX_URL, timeout=30) as response:
        payload = json.load(response)
    region_index_url = payload["offers"]["AmazonEC2"]["currentRegionIndexUrl"]
    with urllib.request.urlopen(PUBLIC_AWS_BASE_URL + region_index_url, timeout=30) as response:
        region_payload = json.load(response)
    return PUBLIC_AWS_BASE_URL + region_payload["regions"][region]["currentVersionUrl"]


@lru_cache(maxsize=8)
def fetch_ec2_offer(region: str) -> dict:
    offer_url = fetch_ec2_region_offer_url(region)
    with urllib.request.urlopen(offer_url, timeout=60) as response:
        return json.load(response)


def lookup_linux_ondemand_hourly_cost(instance_type: str, config: CostConfig) -> float:
    if config.aws_region == "us-east-1" and instance_type in FALLBACK_US_EAST_1_LINUX_ON_DEMAND:
        return FALLBACK_US_EAST_1_LINUX_ON_DEMAND[instance_type]
    try:
        payload = fetch_ec2_offer(config.aws_region)
    except Exception:
        if config.aws_region == "us-east-1" and instance_type in FALLBACK_US_EAST_1_LINUX_ON_DEMAND:
            return FALLBACK_US_EAST_1_LINUX_ON_DEMAND[instance_type]
        raise
    products = payload["products"]
    terms = payload["terms"]["OnDemand"]

    sku = None
    for candidate_sku, product in products.items():
        attributes = product.get("attributes", {})
        if product.get("productFamily") != "Compute Instance":
            continue
        if attributes.get("instanceType") != instance_type:
            continue
        if attributes.get("operatingSystem") != config.ec2_operating_system:
            continue
        if attributes.get("tenancy") != config.tenancy:
            continue
        if attributes.get("preInstalledSw") != config.preinstalled_software:
            continue
        if attributes.get("capacitystatus") != config.capacity_status:
            continue
        if attributes.get("licenseModel") != "No License required":
            continue
        sku = candidate_sku
        break

    if sku is None:
        raise ValueError(f"Could not find AWS price for {instance_type} in {config.aws_region}.")

    ondemand_terms = terms[sku]
    for term in ondemand_terms.values():
        for price_dimension in term["priceDimensions"].values():
            price_per_unit = price_dimension["pricePerUnit"]["USD"]
            return float(price_per_unit)
    raise ValueError(f"Could not parse on-demand pricing for {instance_type}.")


def make_estimate(
    workload: str,
    instance_type: str,
    accelerator: str,
    runtime_seconds: float,
    config: CostConfig,
    notes: str = "",
) -> AwsInstanceEstimate:
    hourly_cost = lookup_linux_ondemand_hourly_cost(instance_type, config)
    runtime_hours = runtime_seconds / 3600
    cost = hourly_cost * runtime_hours
    return AwsInstanceEstimate(
        workload=workload,
        instance_type=instance_type,
        accelerator=accelerator,
        hourly_cost_usd=round(hourly_cost, 6),
        estimated_runtime_hours=round(runtime_hours, 6),
        estimated_cost_usd=round(cost, 6),
        notes=notes,
    )


def ratio_against_minimum(costs: list[float]) -> list[str]:
    if not costs:
        return []
    floor = min(costs)
    if floor <= 0:
        return ["n/a" for _ in costs]
    return [f"{math.ceil(value / floor)}x" for value in costs]
