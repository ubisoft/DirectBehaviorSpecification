using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using UnityEngine.UI;

public class ArenaAgent : Agent
{
    private Camera camera;

    // Agent
    private Rigidbody agentRigidBody;
    private Animator agentAnimator;
    private Renderer agentBodyRenderer;
    
    // Arena components
    [SerializeField] private GameObject goalTile;
    private Renderer goalTileRenderer;
    private Transform goalTileTransform;
    [SerializeField] private GameObject startTile;
    private Renderer startTileRenderer;
    private Transform startTileTransform;
    [SerializeField] private bool lookAtIsActive = true;
    [SerializeField] private GameObject lookAtMarker;
    private Renderer lookAtMarkerHeadRenderer;
    private Transform lookAtMarkerTransform;
    [SerializeField] private GameObject arenaFloor;
    private Renderer arenaFloorRenderer;
    private Transform arenaFloorTransform;
    [SerializeField] private GameObject agentFOV;
    private LineRenderer agentFOVlineRenderer;
    private Transform agentFOVtransform;
    Vector3[] fovPositions = new Vector3[3];

    // Energy bar
    [SerializeField] private Canvas energyBarCanvas;
    [SerializeField] private Slider energyBarSlider;
    [SerializeField] private GameObject rechargeHalo;
    [SerializeField] private bool energyBarIsActive = true;
    private bool agentIsRecharging = false;
    private float energy_gain = 0.01f;
    private float energy_loss = 0.015f;
    private float energy_init = 0.25f;

    // Lava field
    [SerializeField] private bool lavaIsActive = true;
    [SerializeField] private LayerMask lavaLayerMask;
    [SerializeField] private GameObject lavaTilePrefab;
    [SerializeField, Range(10, 1000)] private int n_potential_lava_locations = 1000;
    [SerializeField, Range(-0.1f, 1f)] private float pNoise_lava_theshold = 0.3f;

    GameObject[] lava_tiles;

    private int n_x_points;
    private int n_z_points;

    private float origin_x;
    private float origin_z;

    private float step_size_x;
    private float step_size_z;

    private float[] perlin_array;

    [SerializeField, Range(0, 1000)] private float pNoise_xOrg = 20f;
    [SerializeField, Range(0, 1000)] private float pNoise_zOrg = 20f;
    [SerializeField, Range(1, 10)] private float pNoise_scale = 3.5f;

    private bool touchingLava = false;

    // Parameters
    [SerializeField, Range(1, 50)] private float maxMoveSpeed = 20f;
    [SerializeField, Range(1, 5)] private float maxLookSpeed = 1f;
    [SerializeField, Range(1, 30)] private float maxJumpForce = 15f;
    [SerializeField, Range(1f, 20f)] private float timeScale = 1f;
    [SerializeField] private bool overwriteTimeScale = false;
    [SerializeField] private bool useLaForgeCharacter;
    private int agentSpeedHash = Animator.StringToHash("Speed");
    private int agentJumpHash = Animator.StringToHash("Jump");
    private float arenaSizeX;
    private float arenaSizeZ;
    private float inTheAirThreshold = 0.1f;

    // Dense reward
    [SerializeField] private bool addDenseReward = true;
    Vector3 prev_position;

    // Monitoring
    int step = 0;
    int ep = 0;

    // Side channels
    StringChannel pythonChannel;
    private DecisionRequester decisionRequester;

    public void Awake()
    {
        camera = Camera.main;
        pythonChannel = new StringChannel();
        SideChannelManager.RegisterSideChannel(pythonChannel);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(pythonChannel);
        }
    }

    void Start()
    {
        // Get components

        decisionRequester = GetComponent<DecisionRequester>();

        agentRigidBody = GetComponent<Rigidbody>();
        agentBodyRenderer = GetComponentInChildren<Renderer>();

        if (useLaForgeCharacter){
            agentAnimator = GetComponent<Animator>();
        }

        goalTileRenderer = goalTile.GetComponent<Renderer>();
        goalTileTransform = goalTile.GetComponent<Transform>();

        startTileRenderer = startTile.GetComponent<Renderer>();
        startTileTransform = startTile.GetComponent<Transform>();

        lookAtMarkerHeadRenderer = lookAtMarker.GetComponentInChildren<Renderer>();
        lookAtMarkerTransform = lookAtMarker.GetComponent<Transform>();

        agentFOVlineRenderer = agentFOV.GetComponent<LineRenderer>();
        agentFOVtransform = agentFOV.GetComponent<Transform>();

        pythonChannel.oldLookAtIsActive = lookAtIsActive;
        pythonChannel.newLookAtIsActive = lookAtIsActive;

        this.arenaFloorRenderer = arenaFloor.GetComponent<Renderer>();
        this.arenaFloorTransform = arenaFloor.GetComponent<Transform>();

        // Other pre-computations

        arenaSizeX = this.arenaFloorRenderer.bounds.size.x;
        arenaSizeZ = this.arenaFloorRenderer.bounds.size.z;

        float maxArenaSize = Mathf.Max(arenaSizeX, arenaSizeZ);
        Vector3 fovLimit = this.transform.localPosition + 1.5f * maxArenaSize * this.transform.forward;
        fovPositions[0] = new Vector3(this.transform.localPosition.x, 0.1f, this.transform.localPosition.z);
        fovPositions[1] = new Vector3(fovLimit.x - 0.3f * maxArenaSize, 0.1f, fovLimit.z);
        fovPositions[2] = new Vector3(fovLimit.x + 0.3f * maxArenaSize, 0.1f, fovLimit.z);
        agentFOVlineRenderer.SetPositions(fovPositions);

        // Energy Bar

        pythonChannel.oldEnergyBarIsActive = energyBarIsActive;
        pythonChannel.newEnergyBarIsActive = energyBarIsActive;

        pythonChannel.oldEnergyLoss = energy_loss;
        pythonChannel.newEnergyLoss = energy_loss;

        pythonChannel.oldEnergyGain = energy_gain;
        pythonChannel.newEnergyGain = energy_gain;

        pythonChannel.oldEnergyInit = energy_init;
        pythonChannel.newEnergyInit = energy_init;

        energyBarSlider.gameObject.SetActive(energyBarIsActive);

        // Lava field computations

        // Discretise the plane into n_potential_lava_locations
        float xz_plane_ratio = arenaFloorRenderer.bounds.size.x / arenaFloorRenderer.bounds.size.z;
        n_x_points = Mathf.FloorToInt(Mathf.Sqrt(n_potential_lava_locations / xz_plane_ratio));
        n_z_points = Mathf.FloorToInt(n_potential_lava_locations / n_x_points);

        origin_x = arenaFloorTransform.localPosition.x - arenaFloorRenderer.bounds.size.x / 2f;
        origin_z = arenaFloorTransform.localPosition.z - arenaFloorRenderer.bounds.size.z / 2f;

        step_size_x = arenaFloorRenderer.bounds.size.x / n_x_points;
        step_size_z = arenaFloorRenderer.bounds.size.z / n_z_points;

        // Instantiate the lava tiles
        lava_tiles = new GameObject[n_potential_lava_locations];
        for (int x = 0; x < n_x_points; x++)
        {
            for (int z = 0; z < n_z_points; z++)
            {
                Vector3 location = new Vector3(origin_x + 0.5f * step_size_x + x * step_size_x, 0f, origin_z + 0.5f * step_size_z + z * step_size_z);

                GameObject lavaTile = Instantiate(lavaTilePrefab);
                lavaTile.SetActive(false);
                lava_tiles[x * n_z_points + z] = lavaTile;
                lava_tiles[x * n_z_points + z].transform.localPosition = location;
            }
        }

        pythonChannel.oldLavaIsActive = lavaIsActive;
        pythonChannel.newLavaIsActive = lavaIsActive;
    }

    public override void OnEpisodeBegin()
    {
        ep += 1;
        step = 0;

        // Positioning the lava tiles

        pNoise_xOrg = UnityEngine.Random.Range(0f, 1000f);
        pNoise_zOrg = UnityEngine.Random.Range(0f, 1000f);

        perlin_array = GetPerlinNoise();
        UpdateLavaTilesActivity(perlin_array);

        // Resetting the energy level

        energyBarSlider.value = energy_init;
        agentIsRecharging = false;
        rechargeHalo.SetActive(false);

        // Positioning start-tile, goal-tile, lookAt-marker

        float x_bound = arenaSizeX / 2.2f;
        float z_bound = arenaSizeZ / 2.2f;
        Collider[] hitCollider;

        int k = 0;
        Vector3 startTileInitPos;
        do
        {
            startTileInitPos = new Vector3(UnityEngine.Random.Range(-x_bound, x_bound), 0f, UnityEngine.Random.Range(-z_bound, z_bound));
            hitCollider = Physics.OverlapBox(startTileInitPos, startTileTransform.localScale / 2f, startTileTransform.localRotation, lavaLayerMask);
            k += 1;
        } while (hitCollider.Length != 0
                 && k < 1000
                 );

        startTileTransform.localPosition = startTileInitPos;

        k = 0;
        Vector3 goalTileInitPos;
        do
        {
            goalTileInitPos = new Vector3(UnityEngine.Random.Range(-x_bound, x_bound), 0f, UnityEngine.Random.Range(-z_bound, z_bound));
            hitCollider = Physics.OverlapBox(goalTileInitPos, goalTileTransform.localScale / 2f, goalTileTransform.localRotation, lavaLayerMask);
            k += 1;
        } while ((XZdistance(goalTileInitPos, startTileTransform.localPosition) < 3f * startTileRenderer.bounds.size.x
                 || hitCollider.Length != 0)
                 && k < 1000
                 );

        goalTileTransform.localPosition = goalTileInitPos;

        k = 0;
        Vector3 lookAtMarkerInitPos;
        do
        {
            lookAtMarkerInitPos = new Vector3(UnityEngine.Random.Range(-x_bound, x_bound), lookAtMarkerTransform.localPosition.y, UnityEngine.Random.Range(-z_bound, z_bound));
            k += 1;
        } while ((XZdistance(lookAtMarkerInitPos, startTileTransform.localPosition) < 3f * startTileRenderer.bounds.size.x ||
                  XZdistance(lookAtMarkerInitPos, goalTileTransform.localPosition) < 3f * goalTileRenderer.bounds.size.x) && k < 1000);
        lookAtMarkerTransform.localPosition = lookAtMarkerInitPos;

        // Positioning agent

        this.transform.localPosition = new Vector3(startTileTransform.localPosition.x, 0f, startTileTransform.localPosition.z);
        this.transform.localRotation = Quaternion.LookRotation(new Vector3(lookAtMarkerTransform.localPosition.x - this.transform.localPosition.x, 0f, lookAtMarkerTransform.localPosition.z - this.transform.localPosition.z));   

        agentRigidBody.angularVelocity = Vector3.zero;
        agentRigidBody.velocity = Vector3.zero;

        if (addDenseReward){
            prev_position = this.transform.localPosition;
        }

        if (overwriteTimeScale)
        {
            // Adjust timeScale according to slider
            // mlagents uses a timescale of 20f at training and of 1f when running heuristics
            Time.timeScale = timeScale;

            // Note: Unity Manual recommends lowering Time.fixedDeltaTime when lowering Time.timeScale below 1
            // However, in our use-cases we only increase Time.timeScale (to accelerate training)
            // https://docs.unity3d.com/ScriptReference/Time-timeScale.html?_ga=2.34083170.2116299693.1614011321-862675046.1612454078
        }

        // The DecisionRequester's DecisionPeriod can be modified by sending a message from the python side
        if (pythonChannel.newDecisionPeriod != pythonChannel.oldDecisionPeriod)
        {
            decisionRequester.DecisionPeriod = pythonChannel.newDecisionPeriod;
            pythonChannel.oldDecisionPeriod = pythonChannel.newDecisionPeriod;
            pythonChannel.SendStringToPython(String.Format("DecisionPeriod updated to {0}", pythonChannel.newDecisionPeriod));
        }

        // The lavaIsActive flag can be modified from the python side
        if (pythonChannel.oldLavaIsActive != pythonChannel.newLavaIsActive)
        {
            lavaIsActive = pythonChannel.newLavaIsActive;
            pythonChannel.oldLavaIsActive = pythonChannel.newLavaIsActive;
            pythonChannel.SendStringToPython(String.Format("LavaIsActive updated to {0}", pythonChannel.newLavaIsActive));
        }

        // The lookAtIsActive flag can be modified from the python side
        if (pythonChannel.oldLookAtIsActive != pythonChannel.newLookAtIsActive)
        {
            lookAtIsActive = pythonChannel.newLookAtIsActive;
            pythonChannel.oldLookAtIsActive = pythonChannel.newLookAtIsActive;
            pythonChannel.SendStringToPython(String.Format("LookAtIsActive updated to {0}", pythonChannel.newLookAtIsActive));

            lookAtMarker.SetActive(lookAtIsActive);
            agentFOV.SetActive(lookAtIsActive);
        }

        // The energyBarIsActive flag and energyloss/gain can be modified from the python side
        if (pythonChannel.oldEnergyBarIsActive != pythonChannel.newEnergyBarIsActive)
        {
            energyBarIsActive = pythonChannel.newEnergyBarIsActive;
            pythonChannel.oldEnergyBarIsActive = pythonChannel.newEnergyBarIsActive;
            pythonChannel.SendStringToPython(String.Format("EnergyBarIsActive updated to {0}", pythonChannel.newEnergyBarIsActive));

            energyBarSlider.gameObject.SetActive(energyBarIsActive);
        }

        if (pythonChannel.oldEnergyLoss != pythonChannel.newEnergyLoss)
        {
            energy_loss = pythonChannel.newEnergyLoss;
            pythonChannel.oldEnergyLoss = pythonChannel.newEnergyLoss;
            pythonChannel.SendStringToPython(String.Format("EnergyLoss updated to {0}", pythonChannel.newEnergyLoss));
        }

        if (pythonChannel.oldEnergyGain != pythonChannel.newEnergyGain)
        {
            energy_gain = pythonChannel.newEnergyGain;
            pythonChannel.oldEnergyGain = pythonChannel.newEnergyGain;
            pythonChannel.SendStringToPython(String.Format("EnergyGain updated to {0}", pythonChannel.newEnergyGain));
        }

        if (pythonChannel.oldEnergyInit != pythonChannel.newEnergyInit)
        {
            energy_init = pythonChannel.newEnergyInit;
            pythonChannel.oldEnergyInit = pythonChannel.newEnergyInit;
            pythonChannel.SendStringToPython(String.Format("EnergyInit updated to {0}", pythonChannel.newEnergyInit));
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Look-at color signal

        Vector3[] current_fovWorldPositions = new Vector3[3];
        agentFOVlineRenderer.GetPositions(current_fovWorldPositions);
        for (int i = 0; i < 3; i++) {
            current_fovWorldPositions[i] = agentFOVtransform.TransformPoint(current_fovWorldPositions[i]);
        }

        bool lookAtInAgentView = PointInTriangle(new Vector2(current_fovWorldPositions[0].x, current_fovWorldPositions[0].z),
                                                 new Vector2(current_fovWorldPositions[1].x, current_fovWorldPositions[1].z),
                                                 new Vector2(current_fovWorldPositions[2].x, current_fovWorldPositions[2].z),
                                                 new Vector2(lookAtMarkerTransform.position.x, lookAtMarkerTransform.position.z));
        if (lookAtInAgentView)
        {
            agentFOVlineRenderer.material.color = Color.cyan;
            lookAtMarkerHeadRenderer.material.color = Color.cyan;
        }
        else
        {
            agentFOVlineRenderer.material.color = Color.gray;
            lookAtMarkerHeadRenderer.material.color = Color.gray;
        }

        // Preparing agent observations

        Vector3 agentVelocityXZ = new Vector3(agentRigidBody.velocity.x, 0f, agentRigidBody.velocity.z);
        float agentVelocityXZnorm = agentVelocityXZ.magnitude;
        agentVelocityXZ.Normalize();

        Vector3 goalRelativeDirectionXZ = new Vector3(goalTileTransform.position.x - this.transform.position.x, 0f, goalTileTransform.position.z - this.transform.position.z);
        float goalRelativeDistanceXZ = goalRelativeDirectionXZ.magnitude / Mathf.Max(arenaSizeX, arenaSizeZ);
        goalRelativeDirectionXZ.Normalize();

        Vector3 markerRelativeDirectionXZ = new Vector3(lookAtMarkerTransform.localPosition.x - this.transform.localPosition.x, 0f, lookAtMarkerTransform.localPosition.z - this.transform.localPosition.z);
        float markerRelativeDistanceXZ = markerRelativeDirectionXZ.magnitude / Mathf.Max(arenaSizeX, arenaSizeZ);
        markerRelativeDirectionXZ.Normalize();

        Vector2 XZforward = new Vector2(this.transform.forward.x, this.transform.forward.z);
        Vector2 XZmarkerRelativeDirection = new Vector2(markerRelativeDirectionXZ.x, markerRelativeDirectionXZ.z);

        bool agentOnTheGround = this.transform.localPosition.y < inTheAirThreshold;

        // Agent position and velocity

        sensor.AddObservation(2f * this.transform.localPosition.x / arenaSizeX);  
        sensor.AddObservation(2f * this.transform.localPosition.z / arenaSizeZ);

        sensor.AddObservation(agentVelocityXZ.x);
        sensor.AddObservation(agentVelocityXZ.z);
        sensor.AddObservation(agentVelocityXZnorm / maxMoveSpeed);

        sensor.AddObservation(this.transform.localPosition.y / 2f);
        sensor.AddObservation(Convert.ToSingle(agentOnTheGround));
        sensor.AddObservation(agentRigidBody.velocity.y / maxJumpForce);

        sensor.AddObservation(this.transform.forward.x);
        sensor.AddObservation(this.transform.forward.z);
        sensor.AddObservation(agentRigidBody.angularVelocity.y / maxLookSpeed);

        // Goal tile

        sensor.AddObservation(goalRelativeDirectionXZ.x);
        sensor.AddObservation(goalRelativeDirectionXZ.z);
        sensor.AddObservation(goalRelativeDistanceXZ);
        
        // Look-at Marker

        sensor.AddObservation(markerRelativeDirectionXZ.x);
        sensor.AddObservation(markerRelativeDirectionXZ.z);
        sensor.AddObservation(markerRelativeDistanceXZ);

        sensor.AddObservation(Vector2.SignedAngle(XZforward, XZmarkerRelativeDirection) / 180f);
        sensor.AddObservation(Convert.ToSingle(lookAtInAgentView));

        // Energy bar

        sensor.AddObservation(agentIsRecharging);
        sensor.AddObservation(energyBarSlider.value);

        // Lava tiles

        sensor.AddObservation(Convert.ToSingle(touchingLava));

        RaycastHit hit;
        bool[] raycast_lava_hit_status = new bool[25];
        float ray_shift_size = 1.5f;
        Vector3 ray_dir = new Vector3(0f, -1f, 0f);
        float max_hit_dist = 100f * this.transform.localScale.y;
        int k = 0;
        for (int i = -2; i < 3; i++) 
        {
            for (int j = -2; j < 3; j++) 
            {
                Vector3 ray_start_point = new Vector3(this.transform.localPosition.x + ray_shift_size * i, 10f * this.transform.localScale.y, this.transform.localPosition.z + ray_shift_size * j);
                bool ray_has_hit_lava = Physics.Raycast(ray_start_point, ray_dir, out hit, max_hit_dist, lavaLayerMask);
                raycast_lava_hit_status[k] = ray_has_hit_lava;
                k++;

                sensor.AddObservation(Convert.ToSingle(ray_has_hit_lava));

                if (ray_has_hit_lava)
                {
                    Debug.DrawRay(ray_start_point, ray_dir * hit.distance, Color.yellow);
                }
                else
                {
                    Debug.DrawRay(ray_start_point, ray_dir * max_hit_dist, Color.black);
                }
            }
        }

        // DEBUG ----

        float[] obs = new float[2];
        obs[0] = Convert.ToSingle(agentIsRecharging);
        obs[1] = Convert.ToSingle(touchingLava);
        Debug.Log(String.Format("[{0}, {1}]: obs_subset={2}", ep, step, String.Join(", ", obs)));

        // RESETING SOME FLAGS ----

        touchingLava = false;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        step += 1;
        // Debug.Log(String.Format("ep={0}, step={1}", ep, step));

        // Get agent action

        Vector3 moveSignal = Vector3.zero;
        Vector3 rotateSignal = Vector3.zero;
        float jumpSignal;
        float rechargeSignal;

        moveSignal.x = actionBuffers.ContinuousActions[0];
        moveSignal.z = actionBuffers.ContinuousActions[1];
        rotateSignal.y = actionBuffers.ContinuousActions[2];
        jumpSignal = actionBuffers.ContinuousActions[3];
        rechargeSignal = actionBuffers.ContinuousActions[4];

        // Clamps actions in [0,1]
        
        moveSignal.x = Mathf.Clamp(moveSignal.x, -1f, 1f);
        moveSignal.z = Mathf.Clamp(moveSignal.z, -1f, 1f);
        rotateSignal.y = Mathf.Clamp(rotateSignal.y, -1f, 1f);
        jumpSignal = Mathf.Clamp(jumpSignal, -1f, 1f);
        rechargeSignal = Mathf.Clamp(rechargeSignal, -1f, 1f);

        // Apply agent's actions to physics components

        agentRigidBody.angularVelocity = rotateSignal * maxLookSpeed;
        energyBarSlider.value -= energy_loss;

        if (rechargeSignal <= 0f)  // can't move or jump when you recharge
        {
            agentIsRecharging = false;
            rechargeHalo.SetActive(false);

            if (this.transform.localPosition.y < inTheAirThreshold && Mathf.Abs(agentRigidBody.velocity.y) < 0.01f)
            {
                Vector3 moveSignalvector = new Vector3(moveSignal.x, 0f, moveSignal.z);
                Vector3 desiredVelocity = maxMoveSpeed * Vector3.ClampMagnitude(moveSignalvector, 1f);
                agentRigidBody.velocity = new Vector3(desiredVelocity.x, agentRigidBody.velocity.y, desiredVelocity.z);
            }

            if (jumpSignal > 0 && this.transform.localPosition.y < inTheAirThreshold && Mathf.Abs(agentRigidBody.velocity.y) < 0.01f)
            {
                Vector3 jumpVector = new Vector3(0f, 1f * maxJumpForce, 0f);
                agentRigidBody.AddForce(jumpVector, ForceMode.Impulse);
            }
        }

        if (rechargeSignal > 0 && this.transform.localPosition.y < inTheAirThreshold && Mathf.Abs(agentRigidBody.velocity.y) < 0.01f)
        {
            agentRigidBody.velocity = Vector3.zero;
            energyBarSlider.value += energy_loss + energy_gain;
            agentIsRecharging = true;
            rechargeHalo.SetActive(true);
        }

        // Things to be done after the physics modifications

        energyBarCanvas.transform.LookAt(this.transform.position + camera.transform.rotation * Vector3.back, camera.transform.rotation * Vector3.up);

        // Dense reward for progressing towards the goal

        float denseReward = 0f;
        if (addDenseReward)
        {
            denseReward = (1f / Mathf.Max(arenaSizeX, arenaSizeZ)) * (
                        - XZdistance(goalTileTransform.localPosition, this.transform.localPosition)
                        + XZdistance(goalTileTransform.localPosition, prev_position)
                      );
            
            SetReward(10f * denseReward);
            prev_position = this.transform.localPosition;
        }

        // Termination for reaching the goal (with reward)

        if (goalTileRenderer.bounds.Intersects(agentBodyRenderer.bounds))
        {
            // Final reward for reaching the goal
            
            SetReward(1.0f);
            EndEpisode();
        }

        // Debug.Log($"Episode = {ep}, Step = {step}");
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Use keyboard arrows to move the agent
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");

        // Use keyboard J and L keys to rotate the agent
        continuousActionsOut[2] = Input.GetAxis("ViewHorizontal");

        // Use keyboard space-bar to make the agent jump
        continuousActionsOut[3] = (float)Input.GetAxis("Jump");

        // Use keyboard K key to make the agent recharge
        continuousActionsOut[4] = (float)Input.GetAxis("Recharge");
    }

    private bool SameSide(Vector2 triangle_p1, Vector2 triangle_p2, Vector2 triangle_p3, Vector2 point)
    {
        Vector3 tp1 = new Vector3(triangle_p1.x, 0f, triangle_p1.y);
        Vector3 tp2 = new Vector3(triangle_p2.x, 0f, triangle_p2.y);
        Vector3 tp3 = new Vector3(triangle_p3.x, 0f, triangle_p3.y);
        Vector3 p = new Vector3(point.x, 0f, point.y);
        Vector3 cp1 = Vector3.Cross(tp2 - tp1, tp3 - tp1);
        Vector3 cp2 = Vector3.Cross(tp2 - tp1, p - tp1);
        return Vector3.Dot(cp1, cp2) >= 0f;
    }

    private bool PointInTriangle(Vector2 triangle_p1, Vector2 triangle_p2, Vector2 triangle_p3, Vector2 point)
    {
        // see triangle-point test
        // https://blackpawn.com/texts/pointinpoly/#:~:text=A%20common%20way%20to%20check,triangle%2C%20otherwise%20it%20is%20not
        return SameSide(triangle_p1, triangle_p2, triangle_p3, point) &&
               SameSide(triangle_p2, triangle_p3, triangle_p1, point) &&
               SameSide(triangle_p3, triangle_p1, triangle_p2, point);
    }

    private float XZdistance(Vector3 point1, Vector3 point2)
    {
        return Mathf.Sqrt(Mathf.Pow(point1.x - point2.x, 2f) + Mathf.Pow(point1.z - point2.z, 2f));
    }
    private float XZspeed(Vector3 v)
    {
        return Mathf.Sqrt(v.x * v.x + v.z * v.z);
    }

    private void UpdateLavaTilesActivity(float[] perlin_array)
    {
        for (int x = 0; x < n_x_points; x++)
        {
            for (int z = 0; z < n_z_points; z++)
            {
                if (perlin_array[(n_z_points - z - 1) * n_x_points + (n_x_points - x - 1)] < pNoise_lava_theshold && lavaIsActive)
                {
                    lava_tiles[x * n_z_points + z].SetActive(true);
                }
                else
                {
                    lava_tiles[x * n_z_points + z].SetActive(false);
                }
            }
        }
    }
    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag == "LavaTile")
        {
            touchingLava = true;
        }
    }
    void OnTriggerStay(Collider other)
    {
        if (other.gameObject.tag == "LavaTile")
        {
            touchingLava = true;
        }
    }

    private float[] GetPerlinNoise()
    {
        float[] perlin_array;
        perlin_array = new float[n_x_points * n_z_points];

        // For each pixel in the texture...
        for (int x = 0; x < n_x_points; x++)
        {
            for (int z = 0; z < n_z_points; z++)
            {
                // Debug.Log(String.Format("(x,z)=({0},{1}). total={2}", x, z, x * n_z_points + z));
                float xCoord = pNoise_xOrg + ((float)x / n_x_points) * pNoise_scale;
                float zCoord = pNoise_zOrg + ((float)z / n_z_points) * pNoise_scale;
                float sample = Mathf.PerlinNoise(xCoord, zCoord);
                perlin_array[z * n_x_points + x] = sample;
            }
        }

        return perlin_array;
    }

    private void Update() 
    {
        if (useLaForgeCharacter){
            agentAnimator.SetFloat(agentSpeedHash, XZspeed(agentRigidBody.velocity) / maxMoveSpeed);
            agentAnimator.SetBool(agentJumpHash, this.transform.localPosition.y > inTheAirThreshold);
        }
    }
}