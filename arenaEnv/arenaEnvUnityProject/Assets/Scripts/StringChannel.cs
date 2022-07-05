using UnityEngine;
using Unity.MLAgents.SideChannels;
using System;
using System.Globalization;

public class StringChannel : SideChannel
{
    public int oldDecisionPeriod = -1;
    public int newDecisionPeriod = -1;

    public bool oldLavaIsActive;
    public bool newLavaIsActive;

    public bool oldLookAtIsActive;
    public bool newLookAtIsActive;

    public bool oldEnergyBarIsActive;
    public bool newEnergyBarIsActive;
    public float oldEnergyLoss;
    public float newEnergyLoss;
    public float oldEnergyGain;
    public float newEnergyGain;
    public float oldEnergyInit;
    public float newEnergyInit;

    public StringChannel()
    {
        ChannelId = new Guid("621f0a93-4f87-11ea-a6bf-722f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        // Debug.Log("Received from Python: " + receivedString);
        // this.SendStringToPython("[DEBUG] Received from Python : " + receivedString);

        if (receivedString.StartsWith("DecisionPeriod=")) {
            newDecisionPeriod = int.Parse(receivedString.Replace("DecisionPeriod=", ""));
        }

        if (receivedString.StartsWith("LavaIsActive=")) {
            newLavaIsActive = bool.Parse(receivedString.Replace("LavaIsActive=", ""));
        }

        if (receivedString.StartsWith("LookAtIsActive=")) {
            newLookAtIsActive = bool.Parse(receivedString.Replace("LookAtIsActive=", ""));
        }

        if (receivedString.StartsWith("EnergyBarIsActive="))
        {
            newEnergyBarIsActive = bool.Parse(receivedString.Replace("EnergyBarIsActive=", ""));
        }

        if (receivedString.StartsWith("EnergyLoss="))
        {
            newEnergyLoss = float.Parse(receivedString.Replace("EnergyLoss=", ""), CultureInfo.InvariantCulture);
        }

        if (receivedString.StartsWith("EnergyGain="))
        {
            newEnergyGain = float.Parse(receivedString.Replace("EnergyGain=", ""), CultureInfo.InvariantCulture);
        }

        if (receivedString.StartsWith("EnergyInit="))
        {
            newEnergyInit = float.Parse(receivedString.Replace("EnergyInit=", ""), CultureInfo.InvariantCulture);
        }
    }

    public void SendStringToPython(string stringToSend)
    {
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }
}