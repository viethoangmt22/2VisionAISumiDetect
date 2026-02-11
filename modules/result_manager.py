def aggregate_results(results):
    for r in results:
        if not r["pass"]:
            return "NG"
    return "OK"
